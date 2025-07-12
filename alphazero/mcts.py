#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Monte Carlo Tree Search (MCTS) implementation for the AlphaZero algorithm.
License: MIT
"""
import numpy as np
import random
import torch
from games import Game

class Node:
    def __init__(self, game: Game, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children = {} # Dictionary of child nodes, key is action tuple (row, col)
        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Mean value

    def is_expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(self, net, c_puct=1.4, num_simulations=250, cache_size=100_000):
        self.net = net
        self.device = next(net.parameters()).device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.cache_size = cache_size
        self.eval_cache = self._init_cache()

    def _expand_root(self, root: Node):
        """
        Expands the root node using the network policy. Adds to eval cache if missing.
        """
        key = self._game_to_key(root.game)
        if key in self.eval_cache:
            policy_tensor, _ = self.eval_cache[key]
        else:
            encoded = root.game.encode(device=self.device).to(dtype=torch.float32)

            if encoded.dim() == 3:
                encoded = encoded.unsqueeze(0)

            with torch.no_grad():
                logits, _ = self.net(encoded)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                entropy = -(probs * probs.log()).sum().item()
                #print(f"[Debug] Root policy entropy: {entropy:.3f}")

            policy_tensor = probs.cpu()
            self.eval_cache[key] = (policy_tensor, 0.0)

        self._expand(root, policy_tensor.numpy())



    def _add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """
        Adds Dirichlet noise to the root node's prior to encourage exploration.
        """
        actions = list(root.children.keys())
        num_actions = len(actions)
        if num_actions == 0:
            return  # nothing to do

        noise = np.random.dirichlet([alpha] * num_actions)
        for i, action in enumerate(actions):
            child = root.children[action]
            original_prior = child.prior
            child.prior = (1 - epsilon) * original_prior + epsilon * noise[i]


    def _init_cache(self):
        from collections import OrderedDict
        return OrderedDict()

    def _ucb_score(self, parent: Node, child: Node) -> float:
        # print(f"[Debug] UCB Score for child with prior {child.prior:.3f}, N={child.N}, Q={child.Q:.3f}, self.c_puct={self.c_puct:.3f}")
        return child.Q + self.c_puct * child.prior * (parent.N ** 0.5) / (1 + child.N)

    def _game_to_key(self, game: Game):
        # Flattened CPU float32 view of the state
        return tuple(game.encode(device='cpu').view(-1).tolist())

    def run(self, game: Game, temperature: float = 1.0, add_exploration_noise: bool = False):
        root = Node(game)
        root_player = game.current_player  # Track for correct perspective later

        # --- Ensure root is expanded ---
        self._expand_root(root)

        # --- Inject Dirichlet noise at root ---
        if add_exploration_noise:
            self._add_dirichlet_noise(root)

        # --- Selection & Expansion ---
        pending = []
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            while node.is_expanded() and not node.game.is_terminal():
                # BUG: UCB selection is not working correctly!!!
                # NOTE: This is extremely subtle - at the beginning of the training, need to break ties at random
                # NOTE: I could add noise instead.
                #action, node = max(
                #    node.children.items(),
                #    key=lambda item: self._ucb_score(path[-1], item[1])
                #)

            
                scores = [self._ucb_score(path[-1], item[1]) for item in node.children.items()]
                max_score = max(scores)

                # Choose randomly among all actions with the max UCB score
                best_indices = [i for i, s in enumerate(scores) if s == max_score]
                chosen_index = random.choice(best_indices)
                action, node = list(node.children.items())[chosen_index]

                #print(f"[Debug] Selected action {action} with UCB score {max_score:.3f}")


                # print(f"Items: {list(node.children.items())}")
                #index = np.random.randint(0, len(node.children))
                #action, node = list(node.children.items())[index]

                #print(f"[Debug] Selected action {action} with UCB score {self._ucb_score(path[-1], node):.3f}")
                # print("Choosing a random action")

                #print(f"[Debug]: UCB Scores: {[self._ucb_score(path[-1], item[1]) for item in node.children.items()]}")
                path.append(node)

            if node.game.is_terminal():
                winner = node.game.get_winner()
                # From root player’s perspective
                if winner == 0:
                    value = 0.0
                elif winner == root_player:
                    value = 1.0
                elif winner == -root_player:
                    value = -1.0
                else:
                    raise ValueError(f"Unexpected winner value: {winner}")
                self._backpropagate(path, value)
            else:
                pending.append((node, path))

        # --- Batched NN Evaluation ---
        states, keys, paths = [], [], []
        for node, path in pending:
            key = self._game_to_key(node.game)
            if key in self.eval_cache:
                policy_tensor, value = self.eval_cache[key]
                self._expand(node, policy_tensor.numpy())
                self._backpropagate(path, value)
            else:
                states.append(node.game.encode(device=self.device).to(dtype=torch.float32))
                keys.append(key)
                paths.append((node, path))

        if states:
            batch = torch.cat(states, dim=0)
            with torch.no_grad():
                logits, values = self.net(batch)
                probs = torch.softmax(logits, dim=1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            for key, p, v, (node, path) in zip(keys, probs, values, paths):
                self.eval_cache[key] = (p.cpu(), v.item())
                self._expand(node, p.cpu().numpy())
                self._backpropagate(path, v.item())

                if len(self.eval_cache) > self.cache_size:
                    self.eval_cache.popitem(last=False)

        # --- Compute final policy ---
        return self._get_policy(root, temperature)


    def _expand(self, node: Node, policy: list):
        for action in node.game.get_legal_actions():
            idx = action[0] * node.game.size + action[1]
            node.children[action] = Node(
                game=node.game.apply_action(action),
                parent=node,
                prior=policy[idx]
            )

    def _backpropagate(self, path: list, value: float):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # Switch perspective

    def _get_policy(self, root: Node, temperature: float):
        visits = torch.tensor([c.N for c in root.children.values()], device=self.device, dtype=torch.float32)
        actions = list(root.children.keys())

        if temperature == 0 or visits.sum() == 0:
            best = actions[visits.argmax().item()]
            pi = torch.zeros(root.game.size**2, dtype=torch.float32, device=self.device)
            idx = best[0] * root.game.size + best[1]
            pi[idx] = 1.0
            return pi.cpu(), best

        counts = visits ** (1.0 / temperature)
        probs = counts / counts.sum()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0)
        probs /= probs.sum()

        pi = torch.zeros(root.game.size**2, dtype=torch.float32, device=self.device)
        for a, p in zip(actions, probs):
            idx = a[0] * root.game.size + a[1]
            pi[idx] = p.item()

        action = actions[torch.multinomial(probs, 1).item()]
        return pi.cpu(), action

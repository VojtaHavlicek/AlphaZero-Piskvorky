#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Monte Carlo Tree Search (MCTS) implementation for the AlphaZero algorithm.
License: MIT
"""

import random

import numpy as np
import torch
import torch.nn
from games import Game

DEFAULT_CACHE_SIZE = 100_000  # Default size for the evaluation cache
DEFAULT_NUM_SIMULATIONS = 250  # Default number of simulations per move
DEFAULT_EXPLORATION_STRENGTH = 1.4  # Default exploration strength for UCB


class Node:
    """
    Represents a node in the MCTS tree.
    """

    def __init__(self, game_state: Game, parent=None, policy_prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.policy_prior = policy_prior  # NOTE: This is the prior probability of choosing this node in the UCB formula
        self.children = {}  # Dictionary of child nodes, key is action tuple (row, col)

        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Mean value = W / N

    def is_expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(
        self,
        game_class: type[Game],
        net: torch.nn.Module,
        exploration_strength=DEFAULT_EXPLORATION_STRENGTH,
        num_simulations=DEFAULT_NUM_SIMULATIONS,
        cache_size=DEFAULT_CACHE_SIZE,
    ):
        self.game_class = game_class  # Type of the game, e.g., TicTacToe
        self.net = net  # Neural network for policy and value estimation
        self.exploration_strength = (
            exploration_strength  # Strength of exploration in UCB formula
        )
        self.num_simulations = num_simulations  # Number of simulations to run per move
        self.cache_size = cache_size  # Maximum size of the evaluation cache

        self.device = next(net.parameters()).device  # NOTE: do I need this?
        self.evaluation_cache = self._init_cache()

    def run(
        self,
        game_state: Game,
        temperature: float = 1.0,
        add_exploration_noise: bool = False,
    ):
        """
        Runs the MCTS from a given game state.

        Args:
            game_state (Game): _description_
            temperature (float, optional): _description_. Defaults to 1.0.
            add_exploration_noise (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # --- Initialize root ---
        root = Node(game_state=game_state)
        root_player = game_state.current_player
        self._expand_root(root)

        if add_exploration_noise:
            self._add_dirichlet_noise(root)

        # --- Simulation phase ---
        pending = []
        for _ in range(self.num_simulations):
            path = self._select_and_expand(root)
            leaf = path[-1]

            if leaf.game_state.is_terminal():
                value = self._evaluate_terminal(leaf.game_state, root_player)
                self._backpropagate(path, value)
            else:
                pending.append((leaf, path))

        # --- Evaluation phase ---
        self._evaluate_and_backpropagate(pending)

        # --- Return policy ---
        return self._get_policy(root, temperature)

    def _select_and_expand(self, root: Node):
        """
        Selects a path from the root node to a leaf node, expanding nodes along the way.

        Args:
            root (Node): _description_

        Returns:
            _type_: _description_
        """
        node = root
        path = [node]
        while node.is_expanded() and not node.game_state.is_terminal():
            node = self._select_child(node)
            path.append(node)
        return path

    def _evaluate_terminal(self, game_state: Game, root_player: int) -> float:
        """
        Evaluates the terminal game state to determine the outcome.

        Args:
            game_state (Game): _description_
            root_player (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            float: _description_
        """
        winner = game_state.get_winner()
        if winner == 0:
            return 0.0
        elif winner == root_player:
            return 1.0
        elif winner == -root_player:
            return -1.0
        else:
            raise ValueError(f"Unexpected winner value: {winner}")

    def _evaluate_and_backpropagate(self, pending):
        """
        Evaluates the game states in the pending list using the policy-value network.

        Args:
            pending (list): A list of tuples where each tuple contains a Node and its path.
            Each Node represents a game state that needs to be evaluated.
            The path is a list of nodes leading to the leaf node.
        """
        states, keys, paths = [], [], []

        for node, path in pending:
            key = self._game_state_to_key(node.game_state)
            if key in self.evaluation_cache:
                policy_tensor, value = self.evaluation_cache[key]
                self._expand(node, policy_tensor.numpy())
                self._backpropagate(path, value)
            else:
                states.append(
                    node.game_state.encode(device=self.device).to(dtype=torch.float32)
                )
                keys.append(key)
                paths.append((node, path))

        if not states:
            return

        self.net.eval()  # Ensure the network is in evaluation mode
        batch = torch.cat(states, dim=0)
        with torch.no_grad():
            logits, values = self.net(batch)
            probs = torch.softmax(logits, dim=1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        for key, p, v, (node, path) in zip(keys, probs, values, paths, strict=False):
            self.evaluation_cache[key] = (p.cpu(), v.item())
            self._expand(node, p.cpu().numpy())
            self._backpropagate(path, v.item())

            if len(self.evaluation_cache) > self.cache_size:
                self.evaluation_cache.popitem(last=False)

    def _expand(self, node: Node, policy: list):
        """
        Expands the node by creating child nodes for each legal action.
        This method uses the provided policy to set the prior probabilities for each child node.
        This is the prior probability of choosing this node.

        Args:
            node (Node): _description_
            policy (list): _description_
        """
        for action in node.game_state.get_legal_actions():
            idx = action[0] * node.game_state.board_size + action[1]
            node.children[action] = Node(
                game_state=node.game_state.apply_action(action),
                parent=node,
                policy_prior=policy[idx],
            )

    def _backpropagate(self, path: list, value: float):
        """
        Backpropagates the value through the path from leaf to root.
        This updates the visit count (N), total value (W), and mean value (Q) for each node in the path.

        Args:
            path (list): _description_
            value (float): _description_
        """
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # Switch perspective

    def _get_policy(self, root: Node, temperature: float):
        """
        Returns the policy distribution over actions from the root node.

        Args:
            root (Node): The root node of the MCTS tree.
            temperature (float): Temperature parameter for softmax distribution.

        Returns:
            tuple: A tuple containing the policy distribution tensor and the selected action.
        """
        visits = torch.tensor(
            [c.N for c in root.children.values()],
            device=self.device,
            dtype=torch.float32,
        )
        actions = list(root.children.keys())

        if temperature == 0 or visits.sum() == 0:
            best = actions[visits.argmax().item()]
            policy = torch.zeros(
                root.game_state.board_size**2, dtype=torch.float32, device=self.device
            )
            idx = best[0] * root.game_state.board_size + best[1]
            policy[idx] = 1.0
            return policy.cpu(), best

        counts = visits ** (1.0 / temperature)
        probs = (
            counts / counts.sum()
            if counts.sum() > 0
            else torch.ones_like(counts) / len(counts)
        )
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0)
        probs = (
            probs / probs.sum()
            if probs.sum() > 0
            else torch.ones_like(probs) / len(probs)
        )

        policy = torch.zeros(
            root.game_state.board_size**2, dtype=torch.float32, device=self.device
        )
        for a, p in zip(actions, probs, strict=False):
            idx = a[0] * root.game_state.board_size + a[1]
            policy[idx] = p.item()

        action = actions[torch.multinomial(probs, 1).item()]
        return policy.cpu(), action

    def _ucb_score(self, node: Node, child: Node) -> float:
        """
        Computes the Upper Confidence Bound (UCB) score for a child node.
        This score balances exploration and exploitation.

        Args:
            node (Node): The parent node.
            child (Node): The child node to evaluate.

        Returns:
            float: The UCB score for the child node.
        """
        return child.Q + self.exploration_strength * child.policy_prior * (
            node.N**0.5
        ) / (1 + child.N)

    def _select_child(self, node: Node) -> Node:
        """
        Selects a child node based on the UCB score.

        Args:
            node (Node): The parent node from which to select a child.

        Returns:
            Node: The selected child node.
        """

        assert node.is_expanded(), "Node must be expanded to select a child."

        children_items = list(node.children.items())
        scores = [self._ucb_score(node, item[1]) for item in children_items]
        max_score = max(scores)

        # Choose randomly among all actions with the max UCB score
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        chosen_index = random.choice(best_indices)
        action, node = list(node.children.items())[chosen_index]

        return node  # Return the selected child node

    def _expand_root(self, root: Node):
        """
        Expands the root node using the network policy. Adds to eval cache if missing.

        Args:
            root (Node): The root node to expand.
        """
        key = self._game_state_to_key(root.game_state)
        if key in self.evaluation_cache:
            policy_tensor, _ = self.evaluation_cache[key]
        else:
            encoded = root.game_state.encode(device=self.device).to(dtype=torch.float32)

            if encoded.dim() == 3:
                encoded = encoded.unsqueeze(0)

            with torch.no_grad():
                logits, _ = self.net(encoded)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                entropy = -(probs * probs.log()).sum().item()
                # print(f"[Debug] Root policy entropy: {entropy:.3f}")

            policy_tensor = probs.cpu()
            self.evaluation_cache[key] = (policy_tensor, 0.0)

        self._expand(root, policy_tensor.numpy())

    def _add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """
        Adds Dirichlet noise to the root node's prior to encourage exploration.
        """
        actions = list(
            root.children.keys()
        )  #  Get all legal actions from the root node
        num_actions = len(actions)
        if num_actions == 0:
            return  # nothing to do

        noise = np.random.dirichlet([alpha] * num_actions)
        for i, action in enumerate(actions):
            child = root.children[action]
            original_prior = child.policy_prior
            child.policy_prior = (1 - epsilon) * original_prior + epsilon * noise[i]

    def _init_cache(self):
        from collections import OrderedDict

        return OrderedDict()

    def _game_state_to_key(self, game_state: Game):
        """
        Converts a game state to a unique key for caching.

        Args:
            game_state (Game): _description_

        Returns:
            _type_: _description_
        """
        return tuple(game_state.encode(device="cpu").view(-1).tolist())

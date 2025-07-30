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
from games import Gomoku
from games import X, O, DRAW

DEFAULT_CACHE_SIZE = 500_000  # Default size for the evaluation cache
DEFAULT_NUM_SIMULATIONS = 10_000  # Default number of simulations per move
DEFAULT_EXPLORATION_STRENGTH = 5  # Default exploration strength for UCB


# --- MCTS Node ---
class Node:
    def __init__(self, state, parent=None, prior=1.0):
        self.state = state            # Game state
        self.parent = parent
        self.prior = prior            # Policy prior (P)
        self.children = {}            # action -> Node
        self.N = 0                    # Visit count
        self.W = 0.0                  # Total value
        self.Q = 0.0                  # Mean value

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, policy, legal_actions):
        """Expand node using policy distribution and legal actions."""
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = Node(
                    state=self.state.apply_action(action),
                    parent=self,
                    prior=float(policy[action[0], action[1]])
                )

    def select(self, c_puct):
        """Select child with max Q + U (PUCT)."""
        return max(
            self.children.items(),
            key=lambda item: item[1].Q + c_puct * item[1].prior * np.sqrt(self.N + 1e-8) / (1 + item[1].N)
        )

    def backup(self, value):
        """Backup value through the path, alternating signs."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(-value)


# --- MCTS Core ---
class MCTS:
    def __init__(self, net, num_simulations=1000, c_puct=1.0,
                 dirichlet_alpha=0.3, dirichlet_weight=0.25):
        self.net = net
        self.device = net.device if hasattr(net, 'device') else torch.device('cpu')
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        print(f"[MCTS] Initialized with {num_simulations} simulations, c_puct={c_puct}")

    def run(self,
            root_state, 
            temperature=1.0, 
            add_root_noise=False):
        
        root = Node(state=root_state)

        # --- Evaluate root ---
        with torch.no_grad():
            logits, value = self.net(root_state.encode(self.device).unsqueeze(0))
            policy = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy().reshape(
                root_state.board_size, 
                root_state.board_size
            )
            value = float(value.item())

        legal_actions = root_state.get_legal_actions()
        policy_masked = np.zeros_like(policy)
        for r, c in legal_actions:
            policy_masked[r, c] = policy[r, c]
        policy_masked /= policy_masked.sum() + 1e-8

        # Add exploration noise
        if add_root_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for (r, c), n in zip(legal_actions, noise):
                policy_masked[r, c] = (1 - self.dirichlet_weight) * policy_masked[r, c] + self.dirichlet_weight * n
            policy_masked /= policy_masked.sum()

        root.expand(policy_masked, legal_actions)

        # --- Run simulations ---
        for _ in range(self.num_simulations):
            node, state = root, root_state.clone()

            # Selection
            while not node.is_leaf() and not state.is_terminal():
                action, node = node.select(self.c_puct)
                state = state.apply_action(action)

            # Evaluate leaf
            if state.is_terminal():
                result = state.get_game_result()
                value = 0 if result == DRAW else (1 if result == state.current_player else -1)
            else:
                with torch.no_grad():
                    logits, value_tensor = self.net(state.encode(self.device).unsqueeze(0))
                    policy = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy().reshape(
                        state.board_size, 
                        state.board_size
                    )
                    value = float(value_tensor.item())

                legal_actions = state.get_legal_actions()
                policy_masked = np.zeros_like(policy)

                for r, c in legal_actions:
                    policy_masked[r, c] = policy[r, c]
                policy_masked /= policy_masked.sum() + 1e-8

                node.expand(policy_masked, legal_actions)

            # Backup
            node.backup(-value)

        # --- Compute action probabilities ---
        visit_counts = np.array([child.N for child in root.children.values()], dtype=np.float32)
        actions = list(root.children.keys())

        # One hot policy vector:
        if temperature == 0:
            best_action = actions[np.argmax(visit_counts)]
            pi = np.zeros(root_state.board_size**2, dtype=np.float32)
            idx = best_action[0] * root_state.board_size + best_action[1]
            pi[idx] = 1.0
        else:
            log_counts = np.log(visit_counts + 1e-8) / temperature
            log_counts -= np.max(log_counts)
            counts_temp = np.exp(log_counts)
            probs = counts_temp / counts_temp.sum()

            pi = np.zeros(root_state.board_size**2, dtype=np.float32)
            for (r, c), p in zip(actions, probs):
                pi[r * root_state.board_size + c] = p
            best_action = actions[np.random.choice(len(actions), p=probs)]

        return torch.tensor(pi, dtype=torch.float32), best_action

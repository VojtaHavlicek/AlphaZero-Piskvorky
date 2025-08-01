#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Monte Carlo Tree Search (MCTS) implementation for the AlphaZero algorithm.
License: MIT
"""

# TODO: make sure that the policy is a W x W array, not a flat vector.
import random
import numpy as np
import torch
import torch.nn
from games import Gomoku
from games import X, O, DRAW
from typing import Callable, Tuple, Mapping

DEFAULT_CACHE_SIZE = 500_000  # Default size for the evaluation cache
DEFAULT_NUM_SIMULATIONS = 10_000  # Default number of simulations per move
DEFAULT_EXPLORATION_STRENGTH = 5  # Default exploration strength for UCB


# TODO: add caching 

# --- MCTS Node ---
class Node:
    """
    Node in the MCTS tree.
    """
    def __init__(self, 
                 state, 
                 parent=None, 
                 prior=1.0):
        self.state = state            # Game state
        self.parent = parent
        self.prior = prior            # Policy prior (P)
        self.children = {}            # action -> Node
        self.N = 0                    # Visit count
        self.W = 0.0                  # Total value
        self.Q = 0.0                  # Mean value. Estimated Q-value of the node.

    def is_leaf(self):
        """
        Check if the node is a leaf node (no children).

        Returns:
            _type_: _description_
        """
        return len(self.children) == 0

    def expand(self, policy, legal_actions):
        """
        Expands node using policy distribution and legal actions.

        Args:
            policy (np.ndarray): Policy distribution over actions.
            legal_actions (list[tuple[int, int]]): List of legal actions in the current state.
        """
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = Node(
                    state=self.state.apply_action(action),
                    parent=self,
                    prior=float(policy[action[0], action[1]])
                )

    def select(self, c_puct):
        """Select child with max Q + U (PUCT)."""

    
        # NOTE: max returns the first element, so we need to shuffle the children first
        return max(
            self.children.items(), # TODO
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
    def __init__(self, 
                 policy_value_fn, # Policy and value function, e.g. a neural network. Maps GameState to (policy, value).
                 num_simulations, 
                 c_puct,
                 dirichlet_alpha=0.3, 
                 dirichlet_weight=0.25):
        self.policy_value_fn = policy_value_fn  # Neural network or policy-value function
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        # print(f"[MCTS] Initialized with {num_simulations} simulations, c_puct={c_puct}")

    def run(self,
            root_state, 
            temperature, 
            add_root_noise=False) -> tuple[np.ndarray, tuple[int, int]]:
      
        root = Node(state=root_state)

        # --- Evaluate root ---
        policy, value = self.policy_value_fn(root_state)  # <--- 
        legal_actions = root_state.get_legal_actions()

        # Add exploration noise
        if add_root_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for (r, c), n in zip(legal_actions, noise, strict=True):
                policy[r, c] = (1 - self.dirichlet_weight) * policy[r, c] + self.dirichlet_weight * n
            
            # NOTE: clip this potentailly 

        root.expand(policy, legal_actions)

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
                legal_actions = state.get_legal_actions()
                policy, value = self.policy_value_fn(state)
                node.expand(policy, legal_actions)

            # Backup
            node.backup(-value)

        # --- Compute action probabilities ---
        visit_counts = np.array([child.N for child in root.children.values()], dtype=np.float32)
        actions = list(root.children.keys())

        # Policy vector
        pi = np.zeros(root_state.board_size**2, dtype=np.float32)

        if len(actions) == 0:
            return torch.tensor(pi, dtype=torch.float32), None
        
        if temperature <= 1e-4:
            temperature = 1e-4  # Avoid division by zero
            #best_action = actions[np.argmax(visit_counts)]
            #idx = best_action[0] * root_state.board_size + best_action[1]
            #pi[idx] = 1.0

        # Softmax over visit counts (policy improvement)
        # Apply temperature scaling 
        log_counts = np.log(visit_counts + 1e-8) / temperature
        log_counts -= np.max(log_counts)
        probs = np.exp(log_counts)
        probs_sum = probs.sum()

        # Numerical errors or zero probabilities -> uniform distribution
        if probs_sum < 1e-8 or np.isnan(probs_sum):
            probs = np.ones_like(probs)/len(probs)
        else:
            probs /= probs_sum

        for (r,c), prob in zip(actions, probs, strict=True):
            idx = r * root_state.board_size + c
            pi[idx] = prob

        best_action = actions[np.random.choice(len(actions), p=probs)]

        return pi, best_action

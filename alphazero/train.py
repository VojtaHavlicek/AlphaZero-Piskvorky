#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Training loop for the engine.
License: MIT
"""

import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from collections import deque 
import numpy as np

# ---
from games import Gomoku

from dataclasses import dataclass


@dataclass
class TrainingExample: 
    state: mx.array
    policy: mx.array
    value: float

# --- Replay Buffer --- 

    
# --- Utility ---
def extract_policy_vector(visit_counts: dict, board_size: int = 8) -> mx.array:
    """
    Count how many times was each action visited during MCTS search. 
    This is used to create a policy vector for training the model.

    Args:
        visit_counts (dict): Dictionary mapping (row, col) to visit count.
        board_size (int): Size of the board (default is 8 for Gomoku).

    Returns: 
        mx.array 
    """
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    total_visits = sum(visit_counts.values())

    if total_visits == 0:
        for row_col in visit_counts:
            idx = row_col[0] * board_size + row_col[1]
            policy[idx] = 1.0 
        policy /= np.sum(policy)  # Normalize to sum to 1
    else:
        for (row, col), count in visit_counts.items():
            idx = row * board_size + col
            policy[idx] = count / total_visits
    return mx.array(policy, dtype=mx.float32)


# --- Self-Play ---
def self_play_game(model, num_simulations=50, game=Gomoku):
    game = game() 
    examples = []
    player = game.get_current_player()

    while not game.is_terminal():
        root = Node(game)
        visit_counts = {} 
        for _ in range(num_simulations):
            node = root 
            while node.is_fully_expanded() and not node.state.is_terminal():
                _, node = node.best_child()

            if not node.state.is_terminal():
                node = node.expand() 
            
            if node is not None:
                input_tensor = node.state.encode().reshape(1,-1)
                _, value = model(input_tensor)
                value = float(value.item())
                node.backpropagate(value)

        for action, child in root.children.items():
            visit_counts[action] = child.visits

        policy = extract_policy_vector(visit_counts, board_size=8)
        state_encoded = game.encode() 
        examples.append(TrainingExample(state_encoded, policy, 0.0))

        action = max(root.children.items(), key=lambda x: x[1].visits)[0]
        game.step(action)

    result = game.get_reward(player)
    for ex in examples: 
        ex.value = result 

    return examples


# --- Training ---
def compute_loss(model, states, policy_targets, value_targets):
    pred_policies, pred_values = model(states)

    policy_loss = mx.mean(nn.losses.cross_entropy(pred_policies, policy_targets))
    value_loss = mx.mean((pred_values.squeeze() - value_targets) ** 2)

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss

def train(model, buffer, epochs=10, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grads_fn = nn.value_and_grad(model, fn=compute_loss)

    for epoch in range(epochs):
        if len(buffer) < batch_size:
            continue

        states, policy_targets, value_targets = buffer.sample_batch(batch_size)
    

        # TODO: Match the API here! 
        

        (loss, policy_loss, value_loss), grads = loss_and_grads_fn(model, states, policy_targets, value_targets)


        optimizer.update(model.parameters(), grads)
        mx.eval(model.parameters(), optimizer.state)

        print(f"Epoch {epoch+1}: loss={float(loss.item()):.4f}, policy={float(policy_loss.item()):.4f}, value={float(value_loss.item()):.4f}")


if __name__ == "__main__":
    model = AZNet(input_dim=64, num_actions=64)
    buffer = ReplayBuffer(capacity=5000)

    for iteration in range(10): 
        print(f"\nSelf-play game {iteration +1}")
        game_data = self_play_game(model, num_simulations=100)
        buffer.add_game(game_data)

        print(f"Training on {len(buffer)} examples")
        train(model, buffer, epochs=5, batch_size=64)
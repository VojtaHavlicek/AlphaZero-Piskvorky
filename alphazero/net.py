#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Neural network to be used with AlphaZero algorithm.
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from constants import BOARD_SIZE


class GomokuNet(nn.Module):
    """
    The idea here is to take as an input the state of the board.

    Two outputs:
    1. A continuous value of the board state v_theta(s_t) in [-1,1], from the perspective of the current player.
    2. A policy, the probability vector over all actions.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, board_size=BOARD_SIZE, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.board_size = board_size  # Size of the Gomoku board (e.g., 5 for 5x5 Gomoku)

        # Using architecture from junxiaosong's implementation of AlphaZero for Gomoku.

        # --- Shared backbone ---
        # Conv2d x 3 
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # --- Policy head ---
        # Conv2D -> BatchNorm -> Fully Connected Layer
        # NOTE: First conv layer goes down 128 -> 4 to coordinates.
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(
            4 * board_size * board_size, board_size * board_size
        )  # Output size is board_size * board_size (e.g., 5x5 = 25 for Gomoku)

        # --- Value head ---
        # Conv2D -> BatchNorm -> Fully Connected Layer -> Fully Connected Layer
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)  # Output size is 2 * board_size * board_size
        self.value_fc2 = nn.Linear(64, 1)
   
    def forward(self, x):
        # --- Shared backbone ---
        out = functional.relu(self.conv1(x))
        out = functional.relu(self.conv2(out))
        out = functional.relu(self.conv3(out))

        # --- Policy head ---
        board_size = self.board_size
        policy = functional.relu(self.policy_conv(out))
        policy = policy.view(-1, 4*board_size*board_size)  # Flatten the output for the fully connected layer
        policy_logits = self.policy_fc(policy)

        # --- Value head ---
        value = functional.relu(self.value_conv(out))
        value = functional.relu(self.value_fc1(value.view(value.size(0), -1)))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value # NOTE: returns raw logits for policy and value in [-1, 1]
    
    


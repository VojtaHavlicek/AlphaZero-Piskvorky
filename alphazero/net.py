#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Neural network to be used with AlphaZero algorithm.
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from games import Gomoku

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)
    

class GomokuNet(nn.Module):
    """
    Use 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, board_size=8, num_blocks=3):
        super().__init__()
        num_channels = board_size * board_size # Number of channels in the network, can be adjusted
        self.board_size = board_size

        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        ) # Initial convolution layer. 3 -> num_channels

        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),  # Output channels for policy head
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.board_size * self.board_size),  # Flatten to board size
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),  # Output channels for value head
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Final output for value head
            nn.Tanh()  # Value output in range [-1, 1]
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value.squeeze(-1)  # Squeeze to remove the last dimension for value output
    

class TicTacToeNet(nn.Module):
    """
    Neural network for TicTacToe game.
    Significantly simpler architecture than GomokuNet. 
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.fully_connected_policy = nn.Linear(in_features = 64 * 3 * 3, out_features=9)  # 9 possible actions in TicTacToe
        self.fully_connected_value = nn.Linear(in_features= 64 * 3 * 3, out_features=1) # Value output

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (_type_): Batch of game state samples. 
            Input tensor with shape (batch_size, 3, 3, 3) for TicTacToe.

        Returns:
            _type_: _description_
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten the tensor

        policy_logits = F.log_softmax(self.fully_connected_policy(x), dim=-1)  # Policy logits
        value = torch.tanh(self.fully_connected_value(x))

        return policy_logits, value  # Return policy logits and value output
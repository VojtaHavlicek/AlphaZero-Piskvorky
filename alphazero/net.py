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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return functional.relu(out + residual)


class GomokuNet(nn.Module):
    """
    The idea here is to take as an input the state of the board.

    Two outputs:
    1. A continuous value of the board state v_theta(s_t) in [-1,1], from the perspective of the current player.
    2. A policy, the probability vector over all actions.

    The network is trained using the examples:
    (s_t, pi_t, z_t) where:
    1. p_t is an estimate of the policy from state s_t
    2. z_t is the final outcome of the game from the perspective of the player at s_t (+1 if they win, -1 if they lose, 0 if draw).

    The network is trained to minimize the loss (excluding regularization):
    l = sum_{t=1}^{T} (v_theta(s_t) - z_t)^2 - pi_t * log(p_theta(s_t))

    NOTE: Othello 4 layer CNN + few feedforward works.

    Architeture:
    - two input planes
    - 2-3 residual blocks (why?)
    - 3x3 convolutional layers
    - 2 heads: policy and value:
    policy head: 25 outputs, softmax activation?
    value head: 1 output (value in [-1, 1])

    Args:
        nn (_type_): _description_
    """

    def __init__(self, board_size=5):
        super().__init__()
        num_channels = 32

        # --- Shared backbone ---
        # Conv2D -> BatchNorm -> ReLU -> 2 Residual Blocks
        self.conv = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.res1 = ResidualBlock(num_channels)
        self.res2 = ResidualBlock(num_channels)

        # --- Policy head ---
        # Conv2D -> BatchNorm -> Fully Connected Layer
        # NOTE: First conv layer goes down 32-> 2 to coordinates.
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size
        )  # Output size is board_size * board_size (e.g., 5x5 = 25 for Gomoku)

        # --- Value head ---
        # Conv2D -> BatchNorm -> Fully Connected Layer -> Fully Connected Layer
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)

        # FC goes 25 -> 64 -> 1
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)  # Output size is

    def forward(self, x):
        # NOTE: x will have shape (batch_size, 2, board_size, board_size)

        # --- Shared backbone ---
        out = functional.relu(self.bn(self.conv(x)))
        out = self.res1(out)
        out = self.res2(out)

        # --- Policy head ---
        policy = functional.relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_fc(policy.view(policy.size(0), -1))

        # --- Value head ---
        value = functional.relu(self.value_bn(self.value_conv(out)))
        value = functional.relu(self.value_fc1(value.view(value.size(0), -1)))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

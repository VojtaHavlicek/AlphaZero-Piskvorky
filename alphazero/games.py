#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Abstract game class and Gomoku implementation for AlphaZero algorithm.
License: MIT
"""

import numpy as np
import torch

# NOTE: encodes using relative perspective: 
# THE CURRENT PLAYER IS ALWAYS IN THE FIRST CHANNEL,
# AND THE OPPONENT IS IN THE SECOND CHANNEL.


# --- Abstract Game Class ---
class Game:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = "X"

    def get_legal_actions(self) -> list:
        """Return a list of legal actions for the current player."""
        pass

    def apply_action(self, action) -> "Game":
        pass

    def is_terminal(self) -> bool:
        pass

    def get_winner(self) -> int:
        pass

    def encode(self, device=None) -> torch.Tensor:
        pass

    def clone(self) -> "Game":
        """Return a deep copy of the game state."""
        new_game = Game(self.board_size)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game


# --- Gomoku ---
class Gomoku(Game):
    def __init__(self, board_size=5, win_length=4):
        super().__init__(board_size)
        self.win_length = win_length
        self._winner = None # TODO: Use string instead 

    def get_legal_actions(self) -> list:
        """Return a list of legal actions for the current player."""
        return [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == 0
        ]

    def apply_action(self, action) -> "Gomoku":
        """
        Applies an action to the game state and switches the current player.
        Returns a new game state with the action applied.
        """
        r, c = action
        if self.board[r][c] != 0:
            raise ValueError("Invalid move")
        new_game = self.clone()
        new_game.board[r][c] = self.current_player
        new_game.current_player *= -1
        return new_game

    def _check_line(self, r, c, dr, dc, player) -> bool:
        """
        Check if there is a winning line for player starting from (r, c) in direction (dr, dc).
        """
        count = 0
        for i in range(self.win_length):
            nr, nc = r + i * dr, c + i * dc
            if (
                0 <= nr < self.board_size
                and 0 <= nc < self.board_size
                and self.board[nr][nc] == player
            ):
                count += 1
            else:
                break
        return count == self.win_length

    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state.
        """
        if self._winner is not None:
            return True

        # Check for a winner
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == 0:
                    continue
                player = self.board[r][c]
                # Check horizontal, vertical, and diagonal lines
                if (
                    self._check_line(r, c, 0, 1, player)  # Horizontal
                    or self._check_line(r, c, 1, 0, player)  # Vertical
                    or self._check_line(r, c, 1, 1, player)  # Diagonal \
                    or self._check_line(r, c, 1, -1, player)
                ):
                    self._winner = player
                    return True
        # Check for a draw
        if all(
            self.board[r][c] != 0
            for r in range(self.board_size)
            for c in range(self.board_size)
        ):
            self._winner = 0
            return True
        return False

    def get_winner(self) -> int:
        """Return the winner of the game."""
        if not self.is_terminal():
            return None
        return self._winner if self._winner is not None else 0

    def encode(self, device="cpu") -> torch.Tensor:
        """
        Encode the game state as a tensor.

        Uses a 2D one hot tensor where:
        - The first channel represents the current player's pieces (1 for player, 0 otherwise).
        - The second channel represents the opponent's pieces (-1 for opponent, 0 otherwise).

        NOTE: SUPER IMPORTANT! 
        THE FIRST CHANNEL IS THE CURRENT PLAYER'S PIECES,
        AND THE SECOND CHANNEL IS THE OPPONENT'S PIECES.
        
        NOTE: I dropped the third channel for empty spaces, as it is not needed for Gomoku.

        MARK: Used in AlphaZero's neural network input!
        """
        encoded = torch.zeros(
            (2, self.board_size, self.board_size), dtype=torch.float32, device=device
        )
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == 1:
                    encoded[0, r, c] = 1.0
                elif self.board[r][c] == -1:
                    encoded[1, r, c] = 1.0
        return encoded.view(
            1, 2, self.board_size, self.board_size
        )  # PyTorch expects [1, C, H, W] format for CNNs

    def clone(self) -> "Gomoku":
        """
        Return a deep copy of the game state.
        """
        new_game = Gomoku(self.board_size, self.win_length)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game._winner = self._winner
        return new_game

    def __repr__(self) -> str:
        """
        String representation of the Gomoku board.

        Returns:
            str: A string representation of the Gomoku board.
        """
        board_str = "\n".join(
            " | ".join(
                "X" if self.board[i][j] == 1 else "O" if self.board[i][j] != 0 else " "
                for j in range(self.board_size)
            )
            for i in range(self.board_size)
        )
        return f"Gomoku(\n{board_str}\n)"


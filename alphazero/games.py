#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Abstract game class and Gomoku implementation for AlphaZero algorithm.
License: MIT
"""

# Note: Multiplayer games? 
import torch
import torch.nn as nn
import numpy as np

# --- Abstract Game Class ---
class Game:
    def __init__(self, board_size):
        self.size = board_size 
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1

    def get_legal_actions(self) -> list:
        """Return a list of legal actions for the current player.""" 
        pass 
    def apply_action(self, action) -> 'Game':
        pass
    def is_terminal(self) -> bool:
        pass
    def get_winner(self) -> int:
        pass
    def encode(self, device=None) -> torch.Tensor:
        pass
    def clone(self) -> 'Game':
        """Return a deep copy of the game state."""
        new_game = Game(self.size)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game


# --- Gomoku ---
class Gomoku(Game):
    def __init__(self, board_size=8, win_length=5):
        super().__init__(board_size)
        self._winner = None
        self.win_length = win_length

    def get_legal_actions(self) -> list:
        """Return a list of legal actions for the current player."""
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
    
    def apply_action(self, action) -> 'Gomoku':
        """Apply an action to the game state."""
        r, c = action
        if self.board[r][c] != 0:
            raise ValueError("Invalid move")
        new_game = self.clone()
        new_game.board[r][c] = self.current_player
        new_game.current_player *= -1
        return new_game
    
    def _check_line(self, r, c, dr, dc, player) -> bool:
        """Check if there is a winning line starting from (r, c) in direction (dr, dc)."""
        count = 0
        for i in range(self.win_length):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                count += 1
            else:
                break
        return count == self.win_length
    

    # TODO: Later! 
    #def symmetries(self, policy:torch.Tensor) -> list:
    #    """
    #    Generate symmetries of the game state and policy.
        
    #    Returns a list of tuples (symmetry, policy) where symmetry is the transformed game state.
    #    """
    #    symmetries = []
    #    for flip in [False, True]:
    #        for rotate in range(4):
    #            new_game = self.clone()
    #            if flip:
    #                new_game.board = [row[::-1] for row in new_game.board]
    #            if rotate > 0:
    #                new_game.board = [list(row) for row in zip(*new_game.board[::-1])]
    #            new_policy = policy.clone()
    #            symmetries.append((new_game, new_policy))
    #    return symmetries
    
   
    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state."""
        if self._winner is not None:
            return True

        # Check for a winner
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    continue
                player = self.board[r][c]
                # Check horizontal, vertical, and diagonal lines
                if (self._check_line(r, c, 0, 1, player) or  # Horizontal
                    self._check_line(r, c, 1, 0, player) or  # Vertical
                    self._check_line(r, c, 1, 1, player) or  # Diagonal \
                    self._check_line(r, c, 1, -1, player)):
                    self._winner = player
                    return True
        # Check for a draw
        if all(self.board[r][c] != 0 for r in range(self.size) for c in range(self.size)):
            self._winner = 0
            return True
        return False
    
    def get_winner(self) -> int:
        """Return the winner of the game."""
        if not self.is_terminal():
            return None
        return self._winner if self._winner is not None else 0
    
    def encode(self, device='cpu') -> torch.Tensor:
        """
        Encode the game state as a tensor.
        
        Uses a 3D one hot tensor where:
        - The first channel represents the current player's pieces (1 for player, 0 otherwise).
        - The second channel represents the opponent's pieces (-1 for opponent, 0 otherwise).
        - The third channel represents empty spaces (0 for empty, 1 otherwise).

        MARK: Used in AlphaZero's neural network input! 
        """
        encoded = torch.zeros((3, self.size, self.size), dtype=torch.float32, device=device)
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 1:
                    encoded[0, r, c] = 1.0
                elif self.board[r][c] == -1:
                    encoded[1, r, c] = 1.0
                else:
                    encoded[2, r, c] = 1.0
        return encoded.view(1, 3, self.size, self.size) # PyTorch expects [1, C, H, W] format for CNNs
    
    
    def clone(self) -> 'Gomoku':
        """Return a deep copy of the game state."""
        new_game = Gomoku(self.size, self.win_length)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game
    

    def __repr__(self) -> str:
        """
        String representation of the Gomoku board.

        Returns:
            str: A string representation of the Gomoku board.
        """
        board_str = "\n".join(
            " | ".join("X" if self.board[i][j] == 1 else "O" if self.board[i][j] != 0 else " " for j in range(self.size))
            for i in range(self.size)
        )
        return f"Gomoku(\n{board_str}\n)"
    
# --- TicTacToe ---
class TicTacToe(Game):
    def __init__(self):
        super().__init__(board_size=3)
        self._winner = None

    def get_legal_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == 0]

    def apply_action(self, action):
        r, c = action
        if self.board[r][c] != 0:
            raise ValueError("Invalid move")
        new_game = self.clone()
        new_game.board[r][c] = self.current_player
        new_game.current_player *= -1
        return new_game

    def _check_win(self, player):
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):  # row
                return True
            if all(self.board[j][i] == player for j in range(3)):  # column
                return True
        if all(self.board[i][i] == player for i in range(3)):      # main diag
            return True
        if all(self.board[i][2 - i] == player for i in range(3)):  # anti diag
            return True
        return False

    def is_terminal(self):
        if self._winner is not None:
            return True
        for player in [1, -1]:
            if self._check_win(player):
                self._winner = player
                return True
        if all(self.board[r][c] != 0 for r in range(3) for c in range(3)):
            self._winner = 0  # draw
            return True
        return False

    def get_winner(self):
        if not self.is_terminal():
            return None
        return self._winner

    def encode(self, device='cpu') -> torch.Tensor:
        """
        Encode the TicTacToe game state as a tensor.
        Returns (1, 3, 3, 3) tensor where:
        [1, C, H, W] format:
        - First dimension: Batch size (1 for single game).
        - First channel: Player 1's pieces (1 for player, 0 otherwise).
        - Second channel: Player -1's pieces (-1 for opponent, 0 otherwise).
        - Third channel: Empty spaces (0 for empty, 1 otherwise).

        Args:
            device (str, optional): _description_. Defaults to 'cpu'.

        Returns:
            torch.Tensor: _description_
        """
        encoded = torch.zeros((3, 3, 3), dtype=torch.float32, device=device)
        for r in range(3):
            for c in range(3):
                # One-hot encoding for TicTacToe
                if self.board[r][c] == 1: 
                    encoded[0, r, c] = 1.0
                elif self.board[r][c] == -1:
                    encoded[1, r, c] = 1.0
                else:
                    encoded[2, r, c] = 1.0
        return encoded.view(1, 3, 3, 3)

    def clone(self):
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game._winner = self._winner
        return new_game

    def symmetries(self, policy: torch.Tensor) -> list:
        """
        Generate all 8 symmetries of the board and corresponding policy.

        Args:
            policy (torch.Tensor): 1D tensor of shape (9,), policy over flattened board.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (state_tensor, policy_tensor) pairs.
        """
        assert policy.shape == (9,), "Policy must be flat (9,) tensor"
        board_tensor = self.encode()[0]  # remove batch dimension: (3, 3, 3)
        board_np = board_tensor.cpu().numpy()
        policy_np = policy.view(3, 3).cpu().numpy()

        symmetries = []
        for k in range(4):  # 0, 90, 180, 270 degrees
            for flip in [False, True]:
                rotated_board = np.rot90(board_np, k, axes=(1, 2)).copy() # rotate height/width
                rotated_policy = np.rot90(policy_np, k).copy()

                if flip:
                    rotated_board = np.flip(rotated_board, axis=2).copy() # horizontal flip (cols)
                    rotated_policy = np.flip(rotated_policy, axis=1).copy()

                state_tensor = torch.tensor(rotated_board, dtype=torch.float32)
                policy_tensor = torch.tensor(rotated_policy.flatten(), dtype=torch.float32)

                symmetries.append((state_tensor.unsqueeze(0), policy_tensor))

        return symmetries


    def __repr__(self):
        symbol = {1: 'X', -1: 'O', 0: ' '}
        return "\n".join(
            " | ".join(symbol[self.board[r][c]] for c in range(3))
            for r in range(3)
        )
    

    


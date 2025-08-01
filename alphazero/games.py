#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Abstract game class and Gomoku implementation for AlphaZero algorithm.
License: MIT
"""
import torch
from typing import List, Tuple
from constants import BOARD_SIZE, WIN_LENGTH

# NOTE: encodes using relative perspective: 
# THE CURRENT PLAYER IS ALWAYS IN THE FIRST CHANNEL,
# AND THE OPPONENT IS IN THE SECOND CHANNEL.

X = "X"
O = "O"
DRAW = "D"

# TODO: Add an abstract game class later 

# --- Gomoku ---
class Gomoku():
    def __init__(self, 
                 board_size=BOARD_SIZE, 
                 win_length=WIN_LENGTH):
        self.board_size = board_size
        if not isinstance(board_size, int):
            print(f"[Gomoku] Invalid board size: {board_size}. Must be an integer.")
            raise ValueError("Board size must be an integer.")
        self.board = [[None for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = X
        self.win_length = win_length
        self.winner = None # TODO: Use string instead 
        self.last_action = None  # Store the last move for the opponent

    #MARK: Board manipulation
    def get_legal_actions(self) -> list[tuple[int, int]]:
        """
        Returns a list of legal actions for the current player.
        
        Returns:
            List of tuples (r,c) for valid actions that a player can take. 
        """
        return [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] is None
        ]
    
    def get_other_player(self, player) -> str:
        """
        Returns the other player.

        Args: 
            player: 'X' or 'O'

        Returns: 
            string 'X' on input 'O' 
        """
        if player not in (X, O):
            raise ValueError(f"Invalid player: {player}. Must be 'X' or 'O'.")
        return O if player == X else X
    

    def apply_action(self, action) -> "Gomoku":
        """
        Applies an action to the game state and switches the current player.
        Returns a new game state with the action applied.

        Args: 
            action: (r,c) a pair of row/col 

        Returns: 
            game: a copy of the game
        """
        r, c = action
        if self.board[r][c] is not None:
            raise ValueError("Invalid move")
        new_game = self.clone()
        new_game.board[r][c] = self.current_player
        new_game.current_player = self.get_other_player(new_game.current_player)
        new_game.last_action = action
        return new_game


    # MARK: GameState encoding 
    def encode(self, device) -> torch.Tensor:
        """
        Returns the current board state as a 2D one-hot encoded tensor.
        All from the perspective of the current player.

        Uses a 2D one hot tensor where:
        - The first channel represents the current player's pieces (1 for player, 0 otherwise).
        - The second channel represents the opponent's pieces (-1 for opponent, 0 otherwise).

        NOTE: The third channel can be used to store the last move of the opponent! 
        NOTE: Fourth channel? Empty for now. Can encode starting player or something else.

        MARK: Used in AlphaZero's neural network input!

        Args: 
            pytorch device: "cpu", "mps" or "cuda"

        Returns:         
            torch.Tensor: encoding of the board state and history as (4, board_size, board_size) tensor.
        """
        encoded = torch.zeros(
            (4, self.board_size, self.board_size), dtype=torch.float32, device=device
        )

        current_player = self.current_player
        opponent = self.get_other_player(current_player)

        # Channels 0,1
        # Encode the current player's pieces in the first channel
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == current_player:
                    encoded[0, r, c] = 1.0
                elif self.board[r][c] == opponent:
                    encoded[1, r, c] = 1.0
        
        # Channel 2 
        # If the last action is not None, we can encode it in the third channel
        if self.last_action is not None:
            r, c = self.last_action
            encoded[2, r, c] = 1.0

        # MARK: Fourth channel is empty for now, can be used for additional information
        return encoded
    

    # MARK: Get game results 
    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state.

        Returns:
            bool: true if terminal, false if not. 
        """
        if self.winner is not None:
            return True

        # Check for a winner
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] is None:
                    continue
                player = self.board[r][c]
                # Check horizontal, vertical, and diagonal lines
                if (
                    self._check_line(r, c, 0, 1, player)  # Horizontal
                    or self._check_line(r, c, 1, 0, player)  # Vertical
                    or self._check_line(r, c, 1, 1, player)  # Diagonal \
                    or self._check_line(r, c, 1, -1, player)
                ):
                    self.winner = player
                    return True
        # Check for a draw
        if all(
            self.board[r][c] is not None
            for r in range(self.board_size)
            for c in range(self.board_size)
        ):
            self.winner = DRAW
            return True
        return False
    

    def get_game_result(self) -> str | None:
        """
        Return the result of the game

        Returns: 
            None if the state is not terminal
            'X' if X wins, 'O' if O wins and DRAW if they draw. 
        """
        if not self.is_terminal():
            return None
        return self.winner if self.winner is not None else DRAW


    # MARK: Cloning and symmetry operations
    def rot90(self) -> "Gomoku":
        """
        Rotate the board 90 degrees clockwise and return a new game state.
        """
        new_game = self.clone()
        new_game.board = [list(row) for row in zip(*new_game.board[::-1], strict=False)]
        return new_game
    
    def flip(self) -> "Gomoku":
        """
        Reflect the board horizontally and return a new game state.
        """
        new_game = self.clone()
        new_game.board = [row[::-1] for row in new_game.board]
        return new_game

    def clone(self) -> "Gomoku":
        """
        Return a deep copy of the game state.
        """
        new_game = Gomoku(self.board_size, self.win_length)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        if self.last_action is not None:
            new_game.last_action = tuple(self.last_action)
        return new_game

    # MARK: This method is used to check if there is a winning line for the current player. 
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

    def __repr__(self) -> str:
        """
        String representation of the Gomoku board.

        Returns:
            str: A string representation of the Gomoku board.
        """
        board_str = "\n".join(
            " | ".join(
                self.board[i][j] if self.board[i][j] is not None else " "
                for j in range(self.board_size)
            )
            for i in range(self.board_size)
        )
        return f"Gomoku(\n{board_str}\n)"


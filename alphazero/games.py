# Note: Multiplayer games? 
import torch
import torch.nn as nn

class Game:
    """
    Abstract class for a game
    """
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



class Gomoku(Game):
    """
    Gomoku / Piskvorky 
    """
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
    

        
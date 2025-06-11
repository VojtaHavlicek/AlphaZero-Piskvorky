import mlx.core as mx

class GameState:
    """
    GameState class to manage the state of the game.

    Interface for game states in a reinforcement learning environment.
    """
    def __init__(self):
        self.is_terminal = False

    def clone(self):
        """
        Returns a deep copy of the current game state.
        """
        return GameState()
    
    def get_current_player(self):
        """
        Returns the current player in the game.
        This method should be overridden in subclasses to return the actual player.
        """
        return None
    
    def actions(self):
        """
        Returns the list of possible actions in the current game state.

        Yield instead of returning a list to allow for lazy evaluation? 
        """
        return []
    
    def is_terminal(self):
        """
        Checks if the game state is terminal.
        This method should be overridden in subclasses to provide actual terminal state logic.
        """
        return self.is_terminal
    
    def get_reward(self, player):
        """
        Returns the reward for the given player in the current game state.
        This method should be overridden in subclasses to provide actual reward logic.
        """
        return 0.0
    
    def encode(self):
        """
        Encodes the game state into a format suitable for input to a neural network.
        This method should be overridden in subclasses to provide actual encoding logic.
        """
        return None

    def __repr__(self):
        return f"GameState()"
    

# --- GAME LIBRARY ---

# --- TIC TAC TOE ---
class TicTacToe(GameState):
    def __init__(self):
        self.board = [0] * 9 
        self.current_player = 1 

    def clone(self):
        clone = TicTacToe() 
        clone.board = self.board[:]
        clone.current_player = self.current_player
        return clone
    
    def get_current_player(self):
        return self.current_player
    
    def actions(self):
        return [i for i in range(9) if self.board[i] == 0]
       

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}: Cell already occupied.")
        

        self.board[action] = self.current_player
        self.current_player *= -1 

    def is_terminal(self):
        return self.get_winner() is not None or all(cell != 0 for cell in self.board)
    

    def get_winner(self):
        # Check rows, columns, and diagonals for a winner
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
            (0, 4, 8), (2, 4, 6)              # Diagonals
        ]
        
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        
        return None
    
    def get_reward(self, player):
        winner = self.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0
    
    def encode(self):
        """
        Encodes the TicTacToe board into a format suitable for input to a neural network.
        The encoding is a 1D array of size 9, where:
        - 1 represents player X's move
        - -1 represents player O's move
        - 0 represents an empty cell
        """
        return mx.array([self.board], dtype=mx.float32)
    
    def __repr__(self):
        board_str = "\n".join(
            " | ".join("X" if self.board[i * 3 + j] == 1 else "O" if self.board[i * 3 + j] != 0 else " " for j in range(3))
            for i in range(3)
        )
        return f"TicTacToe(\n{board_str}\n)"
    

# --- PISKVORKY ---
import copy

class Gomoku(GameState):
    SIZE = 8
    WIN_LENGTH = 5

    def __init__(self):
        super().__init__()
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        self.current_player = 1
        self.terminal = False
        self._winner = None

    def clone(self):
        new_game = Gomoku()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.terminal = self.terminal
        new_game._winner = self._winner
        return new_game

    def get_current_player(self):
        return self.current_player

    def actions(self):
        if self.terminal:
            return []
        return [
            (r, c)
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.board[r][c] == 0
        ]

    def step(self, action):
        r, c = action
        if self.board[r][c] != 0:
            raise ValueError("Invalid move")
        self.board[r][c] = self.current_player
        self.current_player *= -1
        self._winner = self._check_winner()
        self.terminal = self._winner is not None or all(
            self.board[r][c] != 0 for r in range(self.SIZE) for c in range(self.SIZE)
        )

    def is_terminal(self):
        return self.terminal

    def get_reward(self, player):
        if not self.terminal:
            return 0.0
        if self._winner == player:
            return 1.0
        elif self._winner == -player:
            return -1.0
        else:
            return 0.0  # draw or no winner

    def encode(self):
        # Encode board from the *current* player's perspective
        flat = [
            float(cell * self.current_player)
            for row in self.board
            for cell in row
        ]
        return mx.array(flat)

    def _check_winner(self):
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r][c] == 0:
                    continue
                if self._check_five(r, c):
                    return self.board[r][c]
        return None

    def _check_five(self, r, c):
        player = self.board[r][c]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, self.WIN_LENGTH):
                nr, nc = r + i * dr, c + i * dc
                if (
                    0 <= nr < self.SIZE and 0 <= nc < self.SIZE and
                    self.board[nr][nc] == player
                ):
                    count += 1
                else:
                    break
            if count >= self.WIN_LENGTH:
                return True
        return False

    def __repr__(self):
        symbols = {1: "X", -1: "O", 0: "."}
        board_str = "\n".join(
            " ".join(symbols[cell] for cell in row)
            for row in self.board
        )
        return f"Gomoku({self.get_current_player()}):\n{board_str}\n"



class GameState:
    """
    GameState class to manage the state of the game.
    """
    def __init__(self):
        self.is_terminal = False

    def clone(self):
        """
        Returns a deep copy of the current game state.
        """
        return GameState()
    
    def current_player(self):
        """
        Returns the current player in the game.
        This method should be overridden in subclasses to return the actual player.
        """
        return None
    
    def actions(self):
        """
        Returns the list of possible actions in the current game state.

        Yield instead of returning a list to allow for lazy evaluation.
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

    def __repr__(self):
        return f"GameState()"
    
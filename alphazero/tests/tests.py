import pytest

# Sample constants (match what you use in your game)
X = "X"
O = "O"
DRAW = "DRAW"

def label_value(winner, current_player):
    """Implements your current labeling logic."""
    return 0 if winner == DRAW else 1 if current_player == winner else -1

def test_value_labeling():
    # Case 1: Draw → always 0
    assert label_value(DRAW, X) == 0
    assert label_value(DRAW, O) == 0

    # Case 2: Winner is X
    assert label_value(X, X) == 1   # X to move, X wins → good for current player
    assert label_value(X, O) == -1  # O to move, X wins → bad for current player

    # Case 3: Winner is O
    assert label_value(O, O) == 1   # O to move, O wins → good for current player
    assert label_value(O, X) == -1  # X to move, O wins → bad for current player

    print("All value-labeling tests passed!")

if __name__ == "__main__":
    test_value_labeling()

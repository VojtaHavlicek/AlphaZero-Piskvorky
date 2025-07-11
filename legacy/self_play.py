from game import GameState, TicTacToe, Gomoku
from model import AZNet, mcts  # Assuming AZModel is defined in model/az_model.py


if __name__ == "__main__":
    game = Gomoku()

    input_dim = len(game.encode())  # Assuming encode() returns a flattened representation of the board
    num_actions = len(game.actions())  # Number of possible actions

    model = AZNet(input_dim=input_dim, hidden_dim=64, num_actions=num_actions) # Assuming AZModel is defined in model/az_model.py

    while not game.is_terminal():
        print(game)
        print(f"Current player: {game.get_current_player()}")
        
        action = mcts(game, model, num_simulations=200)
        game.step(action)
        print(f"Player {game.get_current_player()} takes action: {action}")
        print(f"Current board state: {game}")

    winner = game.get_winner()

    if winner is not None:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")
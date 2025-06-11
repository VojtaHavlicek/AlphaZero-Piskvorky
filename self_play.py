from game import GameState, TicTacToe, Gomoku
from model import AZNet, mcts  # Assuming AZModel is defined in model/az_model.py


if __name__ == "__main__":
    game = Gomoku()
    model = AZNet()  # Assuming AZModel is defined in model/az_model.py

    while not game.is_terminal:
        print(game)
        print(f"Current player: {game.get_current_player()}")
        
        action = mcts(game, model, num_simulations=50)
        game.step(action)
        print(f"Player {game.get_current_player()} takes action: {action}")
        print(f"Current board state: {game}")

    print(game)
    winner = game.get_winner()

    if winner is not None:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")
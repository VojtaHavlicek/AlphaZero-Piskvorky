import torch
from mcts import MCTS
from tqdm import tqdm
from games import O, X, DRAW


class ModelEvaluator:
    """
    Evaluates a candidate neural network against a baseline network using MCTS.
    """

    def __init__(
        self, 
        game_class, 
        mcts_class=None, 
        mcts_params=None, 
        print_games=False,
        device=None
    ):
        self.game_class = game_class
        self.mcts_params = mcts_params or {"exploration_strength": 1.0, "num_simulations": 100}
        self.print_games = print_games  # Whether to print game states during evaluation
        self.device = device if device is not None else torch.device("cpu")


    def evaluate(self, candidate_net, baseline_net, num_games=20, debug=False):
        candidate_net.eval()
        baseline_net.eval()

        candidate_wins = 0
        baseline_wins = 0
        draws = 0

        for i in tqdm(range(num_games), desc="[Evaluator] Evaluating", ncols=80):
            game = self.game_class()
            mcts_candidate = MCTS(self.game_class, candidate_net, **self.mcts_params)
            mcts_baseline = MCTS(self.game_class, baseline_net, **self.mcts_params)

            candidate_symbol, baseline_symbol = X, O

            if i % 2 == 0:
                game.current_player = candidate_symbol
            else:
                game.current_player = baseline_symbol

            while not game.is_terminal():
                if game.current_player == candidate_symbol:
                    mcts = mcts_candidate
                else:
                    mcts = mcts_baseline
                _, action = mcts.run(game, temperature=0)

                game = game.apply_action(action)
                
            if debug:
                print(game)
            
            winner = game.get_winner()
            if winner == candidate_symbol:
                candidate_wins += 1
                print(f"[Evaluator] Candidate wins! Game {i+1}/{num_games}")
            elif winner == baseline_symbol:
                baseline_wins += 1
                print(f"[Evaluator] Baseline wins! Game {i+1}/{num_games}")
            else:
                draws += 1
                print(f"[Evaluator] Draw! Game {i+1}/{num_games}")

            total = candidate_wins + baseline_wins + draws  # Use 0.5 for draws to balance the win rate calculation
            win_rate = (
                (candidate_wins + 0.5*draws) / total
            )  # Default to 50% if no games were played

            if debug:
                print(
                    f"[Evaluator]: Candidate Win Rate: {win_rate:.2%} (W:{candidate_wins} L:{baseline_wins} D:{draws})"
                )

        return win_rate, {
            "wins": candidate_wins,
            "losses": baseline_wins,
            "draws": draws,
            "total": total,
            "win_rate": win_rate,
        }

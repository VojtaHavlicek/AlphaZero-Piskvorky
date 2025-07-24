import torch
from monte_carlo_tree_search import MCTS
from tqdm import tqdm
from games import O, X, DRAW


class ModelEvaluator:
    """
    Evaluates a candidate neural network against a baseline network using MCTS.
    """

    def __init__(
        self, game_class, mcts_class=None, mcts_params=None, print_games=False
    ):
        self.game_class = game_class
        self.mcts_class = mcts_class if mcts_class is not None else MCTS
        self.mcts_params = mcts_params or {}
        self.print_games = print_games  # Whether to print game states during evaluation

    def evaluate(self, candidate_net, baseline_net, num_games=20, debug=False):
        device = next(candidate_net.parameters()).device
        candidate_net.eval()
        baseline_net.eval()

        candidate_wins = 0
        baseline_wins = 0
        draws = 0

        for i in tqdm(range(num_games), desc="[Evaluator] Evaluating", ncols=80):
            game = self.game_class()
            mcts_candidate = self.mcts_class(self.game_class, candidate_net, **self.mcts_params)
            mcts_baseline = self.mcts_class(self.game_class, baseline_net, **self.mcts_params)

            if i % 2 == 0:
                candidate_symbol, baseline_symbol = X, O
                mcts_candidate = self.mcts_class(self.game_class, candidate_net, **self.mcts_params)
                mcts_baseline = self.mcts_class(self.game_class, baseline_net, **self.mcts_params)
            else:
                candidate_symbol, baseline_symbol = O, X
                mcts_candidate = self.mcts_class(self.game_class, candidate_net, **self.mcts_params)
                mcts_baseline = self.mcts_class(self.game_class, baseline_net, **self.mcts_params)

           
            move_sequence = []

            while not game.is_terminal():
                if game.current_player == candidate_symbol:
                    mcts = mcts_candidate
                else:
                    mcts = mcts_baseline
                _, action = mcts.run(game, temperature=0)

                move_sequence.append(action)
                game = game.apply_action(action)
                # if debug:
                #    print(f"[Evaluator] Move {len(move_sequence)}: {action} by {game.current_player}")

            if debug:
                print(game)
            

            winner = game.get_winner()
            if winner == candidate_symbol:
                candidate_wins += 1
            elif winner == baseline_symbol:
                baseline_wins += 1
            else:
                draws += 1

        total = candidate_wins + baseline_wins
        win_rate = (
            candidate_wins / total if total > 0 else 0.5
        )  # Default to 50% if no games were played
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

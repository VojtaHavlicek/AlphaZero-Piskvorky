import torch
from tqdm import tqdm, trange
from mcts import MCTS  # Assuming MCTS is defined in mcts.py

class ModelEvaluator:
    def __init__(self, game_class, mcts_class=None, mcts_params=None):
        self.game_class = game_class
        self.mcts_class = mcts_class if mcts_class is not None else MCTS
        self.mcts_params = mcts_params or {}

    def evaluate(self, candidate_net, baseline_net, num_games=20):
        """
        Evaluate the candidate model against the baseline using alternating roles.

        Returns:
            win_rate: float (percentage of games candidate wins)
            detailed_results: dict with win/loss/draw counts
        """
        device = next(candidate_net.parameters()).device
        candidate_net.eval()
        baseline_net.eval()

        candidate_wins = 0
        baseline_wins = 0
        draws = 0

        for i in tqdm(range(num_games), desc="[Evaluator] Evaluating", ncols=80):
            game = self.game_class()
            mcts1 = self.mcts_class(candidate_net, **self.mcts_params)
            mcts2 = self.mcts_class(baseline_net, **self.mcts_params)

            player_order = (mcts1, mcts2) if i % 2 == 0 else (mcts2, mcts1)
            game.current_player = 1  # Always start with player 1

            while not game.is_terminal():
                mcts = player_order[0] if game.current_player == 1 else player_order[1]
                _, action = mcts.run(game, temperature=0)
                game = game.apply_action(action)

            winner = game.get_winner()
            if winner == 1:
                if i % 2 == 0:
                    candidate_wins += 1
                else:
                    baseline_wins += 1
            elif winner == -1:
                if i % 2 == 0:
                    baseline_wins += 1
                else:
                    candidate_wins += 1
            else:
                draws += 1

        total = candidate_wins + baseline_wins + draws
        win_rate = candidate_wins / total if total > 0 else 0.0
        print(f"[Evaluator]: Candidate Win Rate: {win_rate:.2%} (W:{candidate_wins} L:{baseline_wins} D:{draws})")
        return win_rate, {
            "wins": candidate_wins,
            "losses": baseline_wins,
            "draws": draws,
            "total": total
        }

import torch
from monte_carlo_tree_search import MCTS  # Assuming MCTS is defined in mcts.py
from tqdm import tqdm


class ModelEvaluator:
    # TODO: implement parallel evaluation
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
            mcts1 = self.mcts_class(self.game_class, candidate_net, **self.mcts_params)
            mcts2 = self.mcts_class(self.game_class, baseline_net, **self.mcts_params)

            game.current_player = 1
            player_order = (mcts1, mcts2) if i % 2 == 0 else (mcts2, mcts1)

            move_sequence = []

            if debug:
                with torch.no_grad():
                    encoded = game.encode().to(device)
                    policy_logits, value = candidate_net(encoded)
                    policy_probs = torch.softmax(policy_logits, dim=1)
                    print("\n[Debug] Initial raw policy probabilities:")
                    print(policy_probs.cpu().numpy().reshape(-1))
                    print(f"[Debug] Initial value prediction: {value.item():.4f}")

            while not game.is_terminal():
                mcts = player_order[0] if game.current_player == 1 else player_order[1]
                policy, action = mcts.run(game, temperature=0)

                if debug:
                    print(
                        f"[Debug] Player {game.current_player} selects action: {action}"
                    )
                    print(f"[Debug] Policy: {policy.cpu().numpy().round(3)}")
                    print(f"[Debug] State after action {action}:\n{game}\n------")

                move_sequence.append(action)
                game = game.apply_action(action)

            if debug:
                print(f"[Debug] Game {i} move sequence: {move_sequence}")
                print(game)
                print("------")

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

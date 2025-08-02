import torch
import numpy as np
from constants import (
    EVAL_EXPLORATION_CONSTANT,
    EVAL_TEMPERATURE,
    NUM_EVAL_SIMULATIONS,
    EVAL_TEMPERATURE_SCHEDULE_HALFTIME,
    X, O
)
from controller import NeuralNetworkController, make_policy_value_fn
from mcts import MCTS
from tqdm import tqdm

def temperature_schedule(move: int) -> float:
    """
    Returns a temperature value based on the move number.
    This is used to control the exploration-exploitation trade-off during evaluation.
    """
    return EVAL_TEMPERATURE*np.exp(-move/EVAL_TEMPERATURE_SCHEDULE_HALFTIME)  # Exponential decay

# NOTE: this is so much faster on CPU than MPS, so we use CPU for evaluation! 
class ModelEvaluator:
    """
    Evaluates a candidate neural network against a baseline network using MCTS.
    """

    def __init__(
        self, 
        game_class, 
        print_games,
        device
    ):
        self.game_class = game_class
        self.print_games = print_games  # Whether to print game states during evaluation
        self.device = device if device is not None else torch.device("cpu")


    def evaluate(self, 
                 candidate_controller:NeuralNetworkController, 
                 baseline_controller:NeuralNetworkController, 
                 num_games=20, 
                 debug=False):
        candidate_controller.net.eval()
        baseline_controller.net.eval()

        candidate_wins = 0
        baseline_wins = 0
        draws = 0

        for i in tqdm(range(num_games), desc="[Evaluator] Evaluating", ncols=80):
            game = self.game_class()
            
            candidate_policy_value_fn = make_policy_value_fn(candidate_controller)
            baseline_policy_value_fn = make_policy_value_fn(baseline_controller)

            mcts_candidate = MCTS(candidate_policy_value_fn, 
                                  num_simulations=NUM_EVAL_SIMULATIONS,
                                  c_puct=EVAL_EXPLORATION_CONSTANT)
            
            mcts_baseline = MCTS(baseline_policy_value_fn,
                                  num_simulations=NUM_EVAL_SIMULATIONS,
                                  c_puct=EVAL_EXPLORATION_CONSTANT)

            candidate_symbol, baseline_symbol = X, O

            if i % 2 == 0:
                game.current_player = candidate_symbol
            else:
                game.current_player = baseline_symbol

            candidate_step = 0 
            baseline_step = 0
            while not game.is_terminal():
                if game.current_player == candidate_symbol:
                    mcts = mcts_candidate
                    step = candidate_step
                else:
                    mcts = mcts_baseline
                    step = baseline_step

                _, action = mcts.run(game, 
                                     temperature=temperature_schedule(step), 
                                     add_root_noise=False)

                game = game.apply_action(action)

                if game.current_player == candidate_symbol:
                    candidate_step += 1
                else:
                    baseline_step += 1
                
            if debug:
                print(game)
            
            result = game.get_game_result()
            if result == candidate_symbol:
                candidate_wins += 1
                print(f"[Evaluator] Candidate wins! Game {i+1}/{num_games}")
            elif result == baseline_symbol:
                baseline_wins += 1
                print(f"[Evaluator] Baseline wins! Game {i+1}/{num_games}")
            else:
                draws += 1
                print(f"[Evaluator] Draw! Game {i+1}/{num_games}")

            total = candidate_wins + baseline_wins + draws 
            win_rate = (
                (candidate_wins + 0.5*draws) / total
            ) 

            if debug:
                print(
                    f"[Evaluator] Candidate Win Rate: {win_rate:.2%} (W:{candidate_wins} L:{baseline_wins} D:{draws})"
                )

        return win_rate, {
            "wins": candidate_wins,
            "losses": baseline_wins,
            "draws": draws,
            "total": total,
            "win_rate": win_rate,
        }

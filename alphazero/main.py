#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Human vs AI game using AlphaZero trained model.
License: MIT
"""

import torch
from controller import NeuralNetworkController, make_policy_value_fn
from games import Gomoku, O, X
from mcts import MCTS
from net import GomokuNet
from promoter import ModelPromoter

GAME_CLASS = Gomoku
GAME_CLASS_NET = GomokuNet

# Question: Human in the loop? 
# Can I play the model and generate new training data?


# "models/best_3x3.pt"
def human_vs_ai(model_path=None, model=None):
    """
    Play a game against an AI using a trained model.

    Args:
        model_path (_type_, optional): _description_. Defaults to None.
        model (_type_, optional): _description_. Defaults to None.
    """
    if model is not None:
        net = model
    elif model_path is not None:
        net = GAME_CLASS_NET()
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
    else:
        try:
            promoter = ModelPromoter(
            model_dir="models", evaluator=None, net_class=GomokuNet, device="cpu")
            net = promoter.get_best_model()
        except FileNotFoundError:
            print(
                "No model found. Please provide a valid model path or load a pre-trained model."
            )
            return

    game = GAME_CLASS()
    mcts = MCTS(policy_value_fn=make_policy_value_fn(
        NeuralNetworkController(net, device="cpu")),  # Use CPU for inference
        c_puct=5.0,  # Exploration constant for MCTS
        num_simulations=150,  # Number of MCTS simulations per move
    )

    print(f"You are playing {GAME_CLASS}")

    while not game.is_terminal():
        print(game)

        if game.current_player == X:
            move = input("Your move (row col): ")

            # TODO: sanitize input

            row, col = map(int, move.strip().split())
            if (row, col) not in game.get_legal_actions():
                print("Illegal move, try again.")
                continue
            game = game.apply_action((row, col))
        else:
            print("🤖 AI is thinking...")
            _, action = mcts.run(game, temperature=0)
            print(f"🤖 AI plays: {action}")
            game = game.apply_action(action)

    print(game)
    winner = game.get_game_result()
    if winner == X:
        print("🎉 You win!")
    elif winner == O:
        print("💀 You lost.")
    else:
        print("🤝 It's a draw.")



if __name__ == "__main__":
    human_vs_ai() 

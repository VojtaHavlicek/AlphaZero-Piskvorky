#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Human vs AI game using AlphaZero trained model.
License: MIT
"""

import torch
import torch.nn as nn
from games import TicTacToe
from net import TicTacToeNet
from monte_carlo_tree_search import MCTS
import torch


# "models/best_3x3.pt"
def human_vs_ai(model_path=None,
                model=None):
    
    if model is not None:
        net = model
    elif model_path is not None:
        net = TicTacToeNet()
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
    else:
        try:
            net = TicTacToeNet()
            net.load_state_dict(torch.load("models/best_3x3.pt", map_location="cpu"))
            net.eval()
        except FileNotFoundError:
            print("No model found. Please provide a valid model path or load a pre-trained model.")
            return
    
    game = TicTacToe()
    mcts = MCTS(game_class=TicTacToe, policy_value_net=net)

    print("You are playing as X (1). Type moves like: 3 4")

    while not game.is_terminal():
        print_board(game)

        if game.current_player == 1:
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

    print_board(game)
    winner = game.get_winner()
    if winner == 1:
        print("🎉 You win!")
    elif winner == -1:
        print("💀 You lost.")
    else:
        print("🤝 It's a draw.")

def print_board(game):
    for r in range(game.size):
        row = ""
        for c in range(game.size):
            val = game.board[r][c]
            if val == 1:
                row += " X"
            elif val == -1:
                row += " O"
            else:
                row += " ."
        print(row)
    print()

if __name__ == "__main__":
    human_vs_ai()


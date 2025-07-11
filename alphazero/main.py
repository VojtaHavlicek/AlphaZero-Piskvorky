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
from games import Gomoku
from alphazero.self_play import BOARD_SIZE, WIN_LENGTH
from games import Gomoku
from net import AlphaZeroNet
from mcts import MCTS
import torch



def human_vs_ai(model_path="models/best_3x3.pt", board_size=BOARD_SIZE):
    net = AlphaZeroNet(board_size=BOARD_SIZE, 
                       num_blocks=3)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    game = Gomoku(board_size, win_length=WIN_LENGTH)
    mcts = MCTS(net, num_simulations=100)

    print("You are playing as X (1). Type moves like: 3 4")

    while not game.is_terminal():
        print_board(game)

        if game.current_player == 1:
            move = input("Your move (row col): ")
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


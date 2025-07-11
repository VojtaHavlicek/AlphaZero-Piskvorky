#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Training loop for the engine.
License: MIT
"""

import torch
from net import GomokuNet
from promoter import ModelPromoter
from replay_buffer import ReplayBuffer
from self_play import run_self_play_parallel, train_network, evaluate_models
from games import Gomoku

 # --- Parameters ---
BOARD_SIZE = 3
WIN_LENGTH = 3
NUM_EPISODES = 100
NUM_SELF_PLAY_GAMES = 250 # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
BATCH_SIZE = 64
MODEL_DIR = "models"

if __name__ == "__main__":

    # Set up multiprocessing for MPS backend
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize network, promoter, and replay buffer
    net = GomokuNet(board_size=BOARD_SIZE).to(device)
    promoter = ModelPromoter(MODEL_DIR)
    buffer = ReplayBuffer()

    # Load the best model if it exists
    best_net = GomokuNet(board_size=BOARD_SIZE).to(device)
    best_net.load_state_dict(net.state_dict())


    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n=== Episode {episode} ===")

        # ---- Self-play games ----
        print("SELF-PLAYING GAMES...")
        data = run_self_play_parallel(best_net, num_games=NUM_SELF_PLAY_GAMES,
                                      board_size=BOARD_SIZE, win_length=WIN_LENGTH)
        
        data = [x for game in data for x in game]
        buffer.add(data)

        print(f"Buffer size: {len(buffer)}")


        # ---- Train ----
        print("TRAINING NETWORK...")
        train_network(net, list(buffer.sample(BATCH_SIZE)), epochs=10) # TODO: understand batch size 
        print("Training finished, evaluating...")

        # ---- Evaluate and promote if better ----
        win_rate = evaluate_models(net, best_net, num_games=NUM_SELF_PLAY_GAMES)
        promoter.promote_if_better(net, best_net, win_rate, metadata={"episode": episode, "buffer_size": len(buffer)})
        if promoter.best_path:
            best_net.load_state_dict(torch.load(promoter.best_path, map_location=device))

        # Ensure the net is on the correct device
        net = net.to(device)
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

from net import TicTacToeNet
from games import TicTacToe # NOTE: It may be worth to pakc the recommended ML model into game metadata.
from self_play import SelfPlayManager
from replay_buffer import ReplayBuffer
from trainer import NeuralNetworkTrainer 
from evaluator import ModelEvaluator
from promoter import ModelPromoter
from trainer import generate_bootstrap_dataset, minimax

# Ultimate TicTacToe implementation: 
# Uses BATCH_SIZE = 2048, 
# KL_DIVERGENCE loss, 
# LR_SCHEDULE = lr_schedule={
#            0: 5e-5,
#            1000: 1e-4,
#            2000: 2e-4,
#            3000: 3e-4,
#            50000: 1e-4,
#            85000: 3e-5,
#        },
#        bnm_schedule={
#            95000: 0.02,
#        },

# https://github.com/arnowaczynski/utttai/blob/main/scripts/train_stage1.py 


# --- Parameters ---
BOARD_SIZE = 3
WIN_LENGTH = 3
NUM_EPISODES = 500
NUM_SELF_PLAY_GAMES = 150 # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
BATCH_SIZE = 64 
NUM_EPOCHS = 10 
EVALUATION_GAMES = 50
BUFFER_CAPACITY = 1_000
BOOTSTRAP = True
MODEL_DIR = "models"


import torch


if __name__ == "__main__":

    # Set up multiprocessing for MPS backend
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize network, promoter, and replay buffer
    net = TicTacToeNet().to(device)
    self_play_manager = SelfPlayManager(net, TicTacToe)  
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    evaluator = ModelEvaluator(TicTacToe)
    promoter = ModelPromoter(model_dir=MODEL_DIR, evaluator=evaluator, net_class=TicTacToeNet)
    trainer = NeuralNetworkTrainer(net, device=device)
    
    
    # Optional: only do this if model has not trained before
    if BOOTSTRAP:
        print("[Bootstrap] Generating minimax dataset...")
        bootstrap_data = generate_bootstrap_dataset(
            game_class=TicTacToe,
            minimax_agent=minimax,
            num_games=100,
            max_depth=5,
        )
        buffer.add(bootstrap_data)


    # Go through the training loop
    for episode in range(1, NUM_EPISODES + 1):
        print(f"EPISODE: {episode} \n---------------------------")

        # ---- Self-play games ----
        data = self_play_manager.generate_self_play(num_games=NUM_SELF_PLAY_GAMES,
                                                    num_workers=4)
        
        print(f"[Buffer]: size: {len(buffer)}")
        buffer.add(data) # Add data to the replay buffer
        print(f"[Buffer]: samples added, current size: {len(buffer)}")

        # ---- Train ----
        print(f"[Trainer] Training on {BATCH_SIZE} examples...")
        examples = buffer.sample_batch(BATCH_SIZE)
        trainer.train(examples, epochs=NUM_EPOCHS)
        print(f"[Trainer] Training complete.")
        print("[Debug] Candidate policy logits (first 5):", net(torch.zeros(1, 3, 3, 3).to(device))[0][0][:5].detach().cpu().numpy())


        # ---- Evaluate and promote if better ----
        best_net = promoter.get_best_model()
        # if models_are_equal(net, best_net):
        #     print("⚠️ Candidate model is identical to the baseline.")
        # else:
        #     print("✅ Models differ — evaluation makes sense.")
        win_rate, metrics = promoter.evaluate_and_maybe_promote(net, num_games=EVALUATION_GAMES, metadata={"episode": episode}, debug=True)

        print()
        print("----- Evaluation complete -----")
        # Optional: Print summary
        print(f"[Summary] Win rate: {win_rate:.2%} | Metrics: {metrics}")


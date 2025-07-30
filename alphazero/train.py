#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Training loop for the engine.
License: MIT
"""

import os
import torch
from tqdm import tqdm
from evaluator import ModelEvaluator
from games import (
    Gomoku,  # NOTE: It may be worth to pakc the recommended ML model into game metadata.
)
from net import GomokuNet
from promoter import ModelPromoter
from replay_buffer import ReplayBuffer
from self_play import SelfPlayManager
from controller import NeuralNetworkController

# Ultimate TicTacToe implementation:
# Uses BATCH_SIZE = 2048,
# KL_DIVERGENCE loss,
# LR_SCHEDULE = 
#        bnm_schedule={
#            95000: 0.02,
#        },

# https://github.com/arnowaczynski/utttai/blob/main/scripts/train_stage1.py


# --- Parameters ---
BOARD_SIZE = 5
WIN_LENGTH = 4

# TODO: Set a training schedule 
# 
# WARMUP PHASE: 
# SELF PLAY: 100-200
# EPOCHS: 3-5
# BATCH SIZE: 1024
# EVAL: 20 games

# GROWTH PHASE: 
#
#
#
#
#
#

# REFINEMENT PHASE:

NUM_EPISODES = 100
NUM_SELF_PLAY_GAMES = 10  # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
NUM_SIMULATIONS = 150  # Number of MCTS simulations per move.
NUM_WORKERS = 6  # Adjust based on your CPU cores.
BATCH_SIZE = 2048
NUM_EPOCHS = 4 # How many epochs to train on each batch of data? 
EVALUATION_GAMES = 50
BATCHES_PER_EPISODE = 5 # How many batches to train on each episode (roughly same as the buffer capacity)
LEARNING_RATE = 1e-4  # Initial learning rate for the optimizer
BUFFER_CAPACITY = 10_000
MODEL_DIR = "models"


if __name__ == "__main__":
    # Set up multiprocessing for MPS backend
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    

    # Initialize network, promoter, and replay buffer
    evaluator = ModelEvaluator(Gomoku)
    promoter = ModelPromoter(
        model_dir=MODEL_DIR, 
        evaluator=evaluator, 
        net_class=GomokuNet
    ) 

    net = promoter.get_best_model() # Load the best model or initialize a new one if no model exists
   
    self_play_manager = SelfPlayManager(net, Gomoku, mcts_params={"num_simulations": NUM_SIMULATIONS})
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    if os.path.exists("buffer.pkl"):
        buffer.load("buffer.pkl")  # Load the replay buffer if it exists
    
    trainer = NeuralNetworkController(
        net=net,
        device=device
    )
    
    number_of_promotions = 0 
    # Go through the training loop
    for episode in range(1, NUM_EPISODES + 1):
        print(f"----------------------------\nEPISODE: {episode}/{NUM_EPISODES} \n---------------------------")

        # ---- Self-play games ----
        data = self_play_manager.generate_self_play(
            num_games=NUM_SELF_PLAY_GAMES,
            num_workers=NUM_WORKERS,  # Adjust number of workers based on your CPU cores
        )

        print(f"[Buffer]: size: {len(buffer)}")
        buffer.add(data)  # Add data to the replay buffer
        print(f"[Buffer]: samples added, current size: {len(buffer)}")

        # ---- Train ----
        print(f"[Trainer] Starting training with {len(buffer)} examples in the buffer.")
        for k in range(BATCHES_PER_EPISODE):
            print(f"[Trainer] {k}/{BATCHES_PER_EPISODE} training episode")
            examples = buffer.sample_batch(BATCH_SIZE)
            trainer.train(examples, epochs=NUM_EPOCHS)
        print("[Trainer] Training complete.")
       

        # ---- Evaluate and promote if better ----
        # best_net = promoter.get_best_model()
        # if models_are_equal(net, best_net):
        #     print("⚠️ Candidate model is identical to the baseline.")
        # else:
        #     print("✅ Models differ — evaluation makes sense.")
        win_rate, metrics, was_promoted = promoter.evaluate_and_maybe_promote(
            net, num_games=EVALUATION_GAMES, metadata={"episode": episode}, debug=True
        )
        if was_promoted:
            number_of_promotions += 1

        print()
        print("----- Evaluation complete -----")
        # Optional: Print summary
        print(f"[Summary] Win rate: {win_rate:.2%} | Metrics: {metrics}")

        #TODO: plot policy and value entropies, losses, etc.

    print(f"[Promoter] Number of promotions during the batch: {number_of_promotions}")
    buffer.save("buffer.pkl")  # Save the replay buffer for future use

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
from games import TicTacToe # NOTE: Pack the recommended ML model into metadata? 
from self_play import SelfPlayManager
from replay_buffer import ReplayBuffer
from trainer import NeuralNetworkTrainer 
from evaluator import ModelEvaluator
from promoter import ModelPromoter


 # --- Parameters ---
BOARD_SIZE = 3
WIN_LENGTH = 3
NUM_EPISODES = 75
NUM_SELF_PLAY_GAMES = 150 # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
BATCH_SIZE = 512
MODEL_DIR = "models"


import torch

def models_are_equal(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """
    Returns True if all parameters in both models are bitwise equal,
    after moving model2's parameters to model1's device.
    """
    device = next(model1.parameters()).device
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for k in state_dict1:
        p1 = state_dict1[k]
        p2 = state_dict2[k].to(device)
        if not torch.equal(p1, p2):
            return False
    return True


if __name__ == "__main__":

    # Set up multiprocessing for MPS backend
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize network, promoter, and replay buffer
    net = TicTacToeNet().to(device)
    self_play_manager = SelfPlayManager(net, TicTacToe)  
    buffer = ReplayBuffer(capacity=5_000)
    evaluator = ModelEvaluator(TicTacToe)
    promoter = ModelPromoter(model_dir=MODEL_DIR, evaluator=evaluator, net_class=TicTacToeNet)
    trainer = NeuralNetworkTrainer(net, device=device)
    

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
        trainer.train(examples, epochs=5)
        print(f"[Trainer] Training complete.")
        print("[Debug] Candidate policy logits (first 5):", net(torch.zeros(1, 3, 3, 3).to(device))[0][0][:5].detach().cpu().numpy())


        # ---- Evaluate and promote if better ----
        best_net = promoter.get_best_model()
        # if models_are_equal(net, best_net):
        #     print("⚠️ Candidate model is identical to the baseline.")
        # else:
        #     print("✅ Models differ — evaluation makes sense.")
        win_rate, metrics = promoter.evaluate_and_maybe_promote(net, num_games=10, metadata={"episode": episode}, debug=False)

        print()
        print("----- Evaluation complete -----")
        # Optional: Print summary
        print(f"[Summary] Win rate: {win_rate:.2%} | Metrics: {metrics}")


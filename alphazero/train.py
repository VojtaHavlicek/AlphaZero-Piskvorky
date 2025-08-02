#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Training loop for the engine.
License: MIT
"""
import os

from constants import (
    BATCH_SIZE,
    BATCHES_PER_EPISODE,
    BUFFER_CAPACITY,
    DEVICE,
    EVALUATION_GAMES,
    MODEL_DIR,
    NUM_EPISODES,
    NUM_EPOCHS,
    NUM_SELF_PLAY_GAMES,
    NUM_SELF_PLAY_SIMULATIONS,
    NUM_WORKERS,
    SELF_PLAY_EXPLORATION_CONSTANT,
)
from controller import NeuralNetworkController
from evaluator import ModelEvaluator
from games import (
    Gomoku,  # NOTE: It may be worth to pakc the recommended ML model into game metadata.
)
from model_loader import ModelLoader
from net import GomokuNet
from promoter import ModelPromoter
from replay_buffer import ReplayBuffer
from self_play import SelfPlayManager

    # Add any other constants you need here


if __name__ == "__main__":
    # Set up multiprocessing for MPS backend
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    device = DEVICE #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Main] Using {device}")
    
    
    model_loader = ModelLoader()  # Initialize the model loader
    net = model_loader.get_best_model() # Load the best model or initialize a new one if no model exists
    
    # Initialize network, promoter, and replay buffer
    evaluator = ModelEvaluator(Gomoku,
        print_games=False,  # Set to True to print game states during evaluation
        device=device
    )
    
    promoter = ModelPromoter(
        model_dir=MODEL_DIR, 
        evaluator=evaluator, 
        net_class=GomokuNet,
        device=device
    ) 

   
   
    controller = NeuralNetworkController(
        net=net,
        device=device
    )
   
    self_play_manager = SelfPlayManager(controller, 
                                        device=device,
                                        mcts_params={"num_simulations": NUM_SELF_PLAY_SIMULATIONS,
                                                     "c_puct": SELF_PLAY_EXPLORATION_CONSTANT})
   
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    if os.path.exists("buffer.pkl"):
        buffer.load("buffer.pkl")  # Load the replay buffer if it exists
    
    
    
    number_of_promotions = 0 
    # Go through the training loop
    for episode in range(1, NUM_EPISODES + 1):
        print(f"----------------------------\nEPISODE: {episode}/{NUM_EPISODES} \n---------------------------")

        # ---- Self-play games ----
        data = self_play_manager.generate_self_play(
            num_games=NUM_SELF_PLAY_GAMES,
            num_workers=NUM_WORKERS,  # Adjust number of workers based on your CPU cores
        )
        print(f"[SelfPlayManager] Generated {len(data)} self-play examples. First entry in data: {data[0]}")

        buffer.extend(data)  # Add data to the replay buffer
        print(f"[Buffer]: samples added, current size: {len(buffer)}")

        # ---- Train ----
        print(f"[Trainer] Starting training with {len(buffer)} examples in the buffer.")
        for k in range(BATCHES_PER_EPISODE):
            print(f"[Trainer] {k}/{BATCHES_PER_EPISODE} training episode")
            examples = buffer.sample_batch(BATCH_SIZE)
            #print(f"[Train]: Examples {examples}")
            controller.train(examples, epochs=NUM_EPOCHS)
        print("[Trainer] Training complete.")
       

        # ---- Evaluate and promote if better ----
        # best_net = promoter.get_best_model()
        # if models_are_equal(net, best_net):
        #     print("⚠️ Candidate model is identical to the baseline.")
        # else:
        #     print("✅ Models differ — evaluation makes sense.")
        win_rate, metrics, was_promoted = promoter.evaluate_and_maybe_promote(
            controller, 
            num_games=EVALUATION_GAMES, 
            metadata={"episode": episode}, 
            debug=True
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

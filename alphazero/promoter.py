#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Keeps track of the best model and promotes a new model if it exceeds a win rate threshold.
License: MIT
"""

import os
from datetime import datetime

import torch
from controller import NeuralNetworkController
from model_loader import ModelLoader


class ModelPromoter:
    def __init__(self, model_dir, evaluator, net_class, device, threshold=0.55):
        self.model_dir = model_dir
        self.evaluator = evaluator
        self.net_class = net_class  # To reinstantiate best model
        self.threshold = threshold
        self.device = device
        os.makedirs(model_dir, exist_ok=True)

    def evaluate_and_maybe_promote(
        self, 
        candidate_controller:NeuralNetworkController, 
        num_games=20, 
        metadata=None, 
        debug=False
    ):
        model_loader = ModelLoader()
        baseline_net = model_loader.get_best_model()
        baseline_controller = NeuralNetworkController(baseline_net, 
                                                      device=self.device)
        win_rate, metrics = self.evaluator.evaluate(
            candidate_controller, 
            baseline_controller, 
            num_games=num_games, 
            debug=debug
        )

        was_promoted = False

        if win_rate > self.threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(candidate_controller.net.state_dict(), model_path)
           
            print(
                f"[Promoter]: ✅ Promoted new model with win rate {win_rate:.2%}: {model_path}"
            )
            was_promoted = True
            self.best_path = model_path
            if metadata:
                print("Metadata:", metadata)
        else:
            print(f"[Promoter]: ❌ Candidate rejected (win rate: {win_rate:.2%})")

        return win_rate, metrics, was_promoted  # Return both for logging etc.

   
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


class ModelPromoter:
    def __init__(self, model_dir, evaluator, net_class, threshold=0.55, device="cpu"):
        self.model_dir = model_dir
        self. evaluator = evaluator
        self.net_class = net_class  # To reinstantiate best model
        self.threshold = threshold
        self.device = device
        os.makedirs(model_dir, exist_ok=True)
        self.best_path = self._find_latest_model()

    def get_best_model(self):
        if self.best_path is None:
            print("[Promoter] Choosen a randomly initialized model.")
            random_net = self.net_class().to(self.device)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(random_net.state_dict(), model_path)
            return random_net
        
        print(f"[Promoter] Choosen {self.best_path} for evaluation.")
        model = self.net_class().to(self.device)
        model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        return model

    def evaluate_and_maybe_promote(
        self, candidate_net, num_games=20, metadata=None, debug=False
    ):
        base_net = self.get_best_model()
        win_rate, metrics = self.evaluator.evaluate(
            candidate_net, base_net, num_games=num_games, debug=debug
        )

        was_promoted = False

        if win_rate > self.threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(candidate_net.state_dict(), model_path)
           
            print(
                f"[Promoter]: ✅ Promoted new model with win rate {win_rate:.2%}: {self.best_path} → {model_path}"
            )
            was_promoted = True
            self.best_path = model_path
            if metadata:
                print("Metadata:", metadata)
        else:
            print(f"[Promoter]: ❌ Candidate rejected (win rate: {win_rate:.2%})")

        return win_rate, metrics, was_promoted  # Return both for logging etc.

    def _find_latest_model(self):
        models = [f for f in os.listdir(self.model_dir) if f.endswith(".pt")]
        if not models:
            return None
        latest = max(
            models, key=lambda f: os.path.getctime(os.path.join(self.model_dir, f))
        )
        return os.path.join(self.model_dir, latest)

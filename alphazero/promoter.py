#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Keeps track of the best model and promotes a new model if it exceeds a win rate threshold.
License: MIT
"""

import torch
import os
from datetime import datetime

class ModelPromoter:
    def __init__(self, model_dir, threshold=0.50):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.best_path = None
        self.threshold = threshold

    def promote_if_better(self, candidate_net, base_net, win_rate, metadata=None):
        if win_rate > self.threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(candidate_net.state_dict(), model_path)
            self.best_path = model_path
            print(f"\n✅ New model promoted: {model_path} (win rate: {win_rate:.2%})")
            if metadata:
                print("Metadata:", metadata)
        else:
            print(f"\n❌ Candidate rejected (win rate: {win_rate:.2%})")
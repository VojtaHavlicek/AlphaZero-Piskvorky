import os
from datetime import datetime

import torch
from constants import DEVICE, MODEL_DIR
from net import GomokuNet


class ModelLoader:
    """
    Manages model loading
    """
    def __init__(self):
        """
        Initializes the ModelLoader, setting the model directory and finding the latest model.
        """
        self.model_dir = MODEL_DIR
        self.best_path = self._find_latest_model()
    

    def get_best_model(self):
        """
        Loads the best model from the model directory.
        If no model is found, initializes a new model.

        Returns:
            torch.nn.Module: The best model loaded or a randomly initialized model.
        """
        if self.best_path is None:
            print("[Model Loader] Choosen a randomly initialized model.")
            random_net = GomokuNet() # Assuming GomokuNet is the neural network class
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(random_net.state_dict(), model_path)
            return random_net.float()  # Ensure the model is in float32 format
        
        print(f"[Model Loader] Choosen {self.best_path} for evaluation.")
        model = GomokuNet()
        model.load_state_dict(torch.load(self.best_path, map_location=DEVICE))
        return model.float()  # Ensure the model is in float32 format 
    

    def _find_latest_model(self):
        """
        Finds the latest model in the model directory based on creation time.

        Returns:
            str: Path to the latest model file, or None if no models are found.
        """
        models = [f for f in os.listdir(self.model_dir) if f.endswith(".pt")]
        if not models:
            return None
        latest = max(
            models, key=lambda f: os.path.getctime(os.path.join(self.model_dir, f))
        )
        return os.path.join(self.model_dir, latest)

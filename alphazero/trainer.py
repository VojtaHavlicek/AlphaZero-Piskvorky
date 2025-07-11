import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples  # list of (state, policy, value)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return state, policy, value


class NeuralNetworkTrainer:
    def __init__(self, net, lr=1e-3, batch_size=64, device='cpu'):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.value_loss_fn = torch.nn.MSELoss()
        self.training_history = []

    def _prepare_data(self, examples):
        """
        Converts raw examples to a DataLoader with proper batching and device management.
        """
        dataset = AlphaZeroDataset(examples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, examples, epochs=10):
        dataloader = self._prepare_data(examples)
        self.net.train()

        print("[Trainer] Training started...")

        for epoch in tqdm(range(epochs), desc=f"Epochs", ncols=80):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0

            for state, policy, value in dataloader:

                # NOTE: this forces floats for now. 
                state = state.to(self.device).float()
                policy = policy.to(self.device).float()
                value = value.to(self.device).float()

                pred_policy, pred_value = self.net(state)

                log_probs = F.log_softmax(pred_policy, dim=1)
                loss_policy = -torch.sum(policy * log_probs) / policy.size(0)
                value = value.view(-1, 1)  # Ensure value is of shape (batch_size, 1)
                loss_value = self.value_loss_fn(pred_value, value)
                loss = loss_policy + loss_value

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()
                
            self.training_history.append({
                "loss": total_loss,
                "policy": total_policy_loss,
                "value": total_value_loss
            })

        print(f"[Trainer] Training finished. Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}")

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)

    def eval(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples  # list of (state, policy, value)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return (
            state.float(),
            policy.float(),
            torch.tensor(value, dtype=torch.float32)
            if not torch.is_tensor(value)
            else value.float(),
        )


class NeuralNetworkTrainer:
    def __init__(self, net, lr=1e-3, batch_size=64, device=None):
        if device is None:
            device = torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device

        self.net = net.to(self.device).to(torch.float32)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.value_loss_fn = torch.nn.MSELoss()
        self.training_history = []

    def _prepare_data(self, examples):
        dataset = AlphaZeroDataset(examples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, examples, epochs=10):
        dataloader = self._prepare_data(examples)
        self.net.train()

        print("[Trainer] Training started...")

        for epoch in tqdm(range(epochs), desc="[Trainer] Epochs", ncols=80):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0

            for state, policy, value in dataloader:
                state = state.to(self.device, dtype=torch.float32)
                policy = policy.to(self.device, dtype=torch.float32)
                value = value.to(self.device, dtype=torch.float32).view(-1, 1)

                pred_policy, pred_value = self.net(state)

                # --- LOSS ---
                log_probs = F.log_softmax(pred_policy, dim=1)
                loss_policy = F.kl_div(log_probs, policy, reduction="batchmean")
                loss_value = self.value_loss_fn(pred_value, value)
                loss = loss_policy + loss_value

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()

            # print(f"[Trainer] Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}")
            self.training_history.append(
                {
                    "loss": total_loss,
                    "policy": total_policy_loss,
                    "value": total_value_loss,
                }
            )

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: Total={total_loss:.4f}, Policy={total_policy_loss:.4f}, Value={total_value_loss:.4f}"
                )

        print(
            f"[Trainer] Training finished. Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}"
        )

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net = self.net.to(self.device).to(torch.float32)

    def eval(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

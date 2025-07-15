import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import random

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
            torch.tensor(value, dtype=torch.float32) if not torch.is_tensor(value) else value.float()
        )


class NeuralNetworkTrainer:
    def __init__(self, net, lr=1e-3, batch_size=64, device=None):
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
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

        for epoch in tqdm(range(epochs), desc=f"[Trainer] Epochs", ncols=80):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0

            for state, policy, value in dataloader:
                state = state.to(self.device, dtype=torch.float32)
                policy = policy.to(self.device, dtype=torch.float32)
                value = value.to(self.device, dtype=torch.float32).view(-1, 1)

                pred_policy, pred_value = self.net(state)

                log_probs = F.log_softmax(pred_policy, dim=1)
                loss_policy = -torch.sum(policy * log_probs) / policy.size(0)
                loss_value = self.value_loss_fn(pred_value, value)
                loss = loss_policy + loss_value

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += loss_policy.item()
                total_value_loss += loss_value.item()

            # print(f"[Trainer] Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}")
            self.training_history.append({
                "loss": total_loss,
                "policy": total_policy_loss,
                "value": total_value_loss
            })

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Total={total_loss:.4f}, Policy={total_policy_loss:.4f}, Value={total_value_loss:.4f}")


        print(f"[Trainer] Training finished. Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}")

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net = self.net.to(self.device).to(torch.float32)

    def eval(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()


import torch
from typing import List, Tuple

def generate_bootstrap_dataset(
    game_class: type,
    minimax_agent,
    num_games: int = 100,
    max_depth: int = 5,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Generate a bootstrap dataset using minimax-vs-minimax games.

    Args:
        game_class: A class implementing the Game interface.
        minimax_agent: A function(game_state, depth, maximizing_player) -> (value, action)
        num_games: Number of games to simulate
        max_depth: Depth for the minimax agent

    Returns:
        List of (state_tensor, policy_tensor, value) tuples

    TODO: add heuristic to the minimax agent to avoid deep recursion
    """
    dataset = []

    for game_index in tqdm(range(num_games), desc="[Bootstrap] Generating", ncols=80):
        game = game_class()
        history = []

        while not game.is_terminal():
            _, action = minimax(game, depth=9, maximizing_player=True, root_player=game.current_player)

            if action is None:
                break  # Defensive: terminal state reached or minimax gave up

            state = game.encode().squeeze(0)  # shape: (3, board_size, board_size)
            size = game.size
            policy = torch.zeros(size * size, dtype=torch.float32)

            idx = action[0] * size + action[1]
            policy[idx] = 1.0

            history.append((state, policy, game.current_player))

           
            game = game.apply_action(action)
            print(f"[Debug] State after action {action}:\n{game}\n------")

            

        # Get winner at the end
        winner = game.get_winner()
        for state, policy, player in history:
            value = 1.0 if winner == player else -1.0 if winner == -player else 0.0
            dataset.append((state, policy, value))

    return dataset


import random

def minimax(game, depth, maximizing_player, root_player):
    if game.is_terminal() or depth == 0:
        winner = game.get_winner()
        if winner == root_player:
            return 1, None
        elif winner == -root_player:
            return -1, None
        else:
            return 0, None

    best_value = float('-inf') if maximizing_player else float('inf')
    best_actions = []

    for action in game.get_legal_actions():
        child = game.apply_action(action)
        val, _ = minimax(child, depth - 1, not maximizing_player, root_player)

        if maximizing_player:
            if val > best_value:
                best_value = val
                best_actions = [action]
            elif val == best_value:
                best_actions.append(action)
        else:
            if val < best_value:
                best_value = val
                best_actions = [action]
            elif val == best_value:
                best_actions.append(action)

    best_action = random.choice(best_actions) if best_actions else None
    return best_value, best_action





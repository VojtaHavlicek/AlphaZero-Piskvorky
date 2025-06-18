import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from mcts import MCTS
from games import Gomoku
from net import AlphaZeroNet

from datetime import datetime
from collections import deque
import random

import shutil, json
from pathlib import Path

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer  # Return all if not enough samples
        return random.sample(self.buffer, batch_size)

    def all(self):
        return list(self.buffer)
    
    def __len__(self):
        return len(self.buffer) 


from torch.utils.data import Dataset 

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return state, policy, torch.tensor([value], dtype=torch.float32)
    

class AlphaZeroTrainer:
    def __init__(self, net, mcts_class, game_class, promoter: 'ModelPromoter', buffer_size=10000):
        self.device = get_best_device()
        self.net = net.to(self.device)
        self.promoter = promoter
        self.mcts_class = mcts_class
        self.game_class = game_class
        self.buffer = ReplayBuffer(max_size=buffer_size)

    def play_self_play_game(self, temperature_threshold=10):
        game = self.game_class()
        mcts = self.mcts_class(self.net)
        history = []
        move_num = 0

        while not game.is_terminal():
            temperature = 1.0 if move_num < temperature_threshold else 0.01
            policy, action = mcts.run(game, temperature=temperature)
            state = game.encode().squeeze(0)
            history.append((state, policy, game.current_player))
            game = game.apply_action(action)
            move_num += 1

        winner = game.get_winner()
        data = [(s, p, 1 if winner == cp else -1 if winner == -cp else 0)
                for (s, p, cp) in history]
        return data

    def train_step(self, batch_size=64, epochs=10, lr=1e-3):
        #if len(self.buffer) < batch_size:
        #    print("Not enough examples to train.")
        #    return

        dataset = AlphaZeroDataset(self.buffer.sample(batch_size))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn_value = torch.nn.MSELoss()

        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for state, policy, value in dataloader:
                state = state.to(self.device)
                policy = policy.to(self.device)
                value = value.to(self.device)
                pred_policy, pred_value = self.net(state)
                loss_policy = -torch.sum(policy * torch.nn.functional.log_softmax(pred_policy, dim=1)) / policy.size(0)
                loss_value = loss_fn_value(pred_value.view(-1, 1), value)
                loss = loss_policy + loss_value
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader):.4f}")

    def evaluate_and_promote(self, candidate_net, eval_fn, metadata=None):
        best_net = self.net.__class__().to(self.device)  # Create a new instance of the net
        best_net.load_state_dict(torch.load(self.promoter.best_path))
        candidate_net = candidate_net.to(self.device)
        winrate = eval_fn(candidate_net, best_net)
        self.promoter.promote_if_better(winrate, candidate_net, metadata)

    def train_loop(self, episodes=10, eval_fn=None):
        for episode in range(episodes):
            print(f"\n🚀 Self-play game {episode+1}")
            game_data = self.play_self_play_game()
            self.buffer.add(game_data)
            self.train_step()

            if eval_fn is not None:
                print("\n⚔️  Evaluating candidate model...")
                self.evaluate_and_promote(self.net, eval_fn, metadata={
                    "episode": episode + 1,
                    "buffer_size": len(self.buffer)
                })



class ModelPromoter:
    def __init__(self, model_dir="models", threshold=0.55):
        self.model_dir = Path(model_dir)
        self.threshold = threshold

        self.best_path = self.model_dir / "best.pt"
        self.snapshots_path = self.model_dir / "snapshots"
        self.logs_path = self.model_dir / "logs"

        self.snapshots_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def promote_if_better(self, winrate, model: torch.nn.Module, metadata: dict = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if winrate > self.threshold:
            print(f"\n✅ Promoting candidate model (win rate = {winrate:.2f})")
            snapshot_path = self.snapshots_path / f"model_{timestamp}.pt"
            torch.save(model.state_dict(), snapshot_path)
            shutil.copy(snapshot_path, self.best_path)
            status = "promoted"
        else:
            print(f"\n❌ Candidate model rejected (win rate = {winrate:.2f})")
            status = "rejected"

        log = {
            "timestamp": timestamp,
            "winrate": winrate,
            "status": status,
            **(metadata or {})
        }
        log_path = self.logs_path / f"model_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        return log


def evaluate_models(candidate_net, best_net, game_class, mcts_class, num_games=20, num_simulations=200, verbose=False):
    candidate_wins = 0
    best_wins = 0
    draws = 0

    for game_idx in range(num_games):
        game = game_class()

        # Alternate who is first player
        if game_idx % 2 == 0:
            players = {1: mcts_class(candidate_net, num_simulations=num_simulations),
                       -1: mcts_class(best_net, num_simulations=num_simulations)}
            first = "Candidate"
        else:
            players = {1: mcts_class(best_net, num_simulations=num_simulations),
                       -1: mcts_class(candidate_net, num_simulations=num_simulations)}
            first = "Best"

        if verbose:
            print(f"\nGame {game_idx+1}: {first} goes first")

        while not game.is_terminal():
            mcts = players[game.current_player]
            _, action = mcts.run(game, temperature=0.0)  # deterministic best move
            game = game.apply_action(action)

        winner = game.get_winner()
        if winner == 1:
            if first == "Candidate":
                candidate_wins += 1
            else:
                best_wins += 1
        elif winner == -1:
            if first == "Candidate":
                best_wins += 1
            else:
                candidate_wins += 1
        else:
            draws += 1

    total = candidate_wins + best_wins + draws
    win_rate = candidate_wins / total if total > 0 else 0.0
    print(f"\n🏁 Evaluation Result: Candidate Win Rate = {win_rate:.2f} ({candidate_wins}W / {best_wins}L / {draws}D)")
    return win_rate


if __name__ == "__main__":
    net = AlphaZeroNet(board_size=8)
    mcts = MCTS(net, num_simulations=1000)
    promoter = ModelPromoter(model_dir="models", threshold=0.55)
    trainer = AlphaZeroTrainer(net, mcts_class=MCTS, game_class=Gomoku, promoter=promoter, buffer_size=10000)

    # Example training loop
    trainer.train_loop(episodes=10, 
                       eval_fn=lambda cnet, bnet: evaluate_models(
        candidate_net=cnet,
        best_net=bnet,
        game_class=Gomoku,
        mcts_class=MCTS,
        num_games=20,
        num_simulations=1000
    ))  # Replace eval_fn with your evaluation function
    

    



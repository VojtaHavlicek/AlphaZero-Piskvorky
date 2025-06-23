import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mcts import MCTS
from games import Gomoku
from net import AlphaZeroNet

from datetime import datetime
from collections import deque
import random

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

BOARD_SIZE = 3  # Default board size for Gomoku
WIN_LENGTH = 3  # Default win length for Gomoku

def play_self_play_game(mcts:MCTS):
    game = Gomoku(size=BOARD_SIZE, win_length=WIN_LENGTH)
    history = [] 
    move_num = 0 # Annealing TODO: optimal schedule? 

    # Play a self-play game using MCTS
    while not game.is_terminal():
        temp = 1.0 if move_num < 10 else 0.01  # Annealing temperature
        policy, action = mcts.run(game, temperature=temp)
        state = game.encode().squeeze(0) 
        history.append((state, policy, game.current_player))
        game = game.apply_action(action)

    # Game is over 
    winner = game.get_winner()

    # Convert history to a format suitable for training
    # NOTE: see that only terminal states are rewarded!
    data = []
    for state, policy, player in history:
        if winner == player:
            value = 1.0
        elif winner == 0:
            value = 0.0
        else:
            value = -1.0
        
        data.append((state, policy, value))
    
    return data # List of tuples (state, policy, value) 



from torch.utils.data import Dataset 

class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return state, policy, torch.tensor([value], dtype=torch.float32)
    


def train_network(net, examples, epochs=5, batch_size=32, lr=1e-3):
    dataset = AlphaZeroDataset(examples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    net.train()
    for epoch in range(epochs):
        total_loss = 0
        for state, policy, value in dataloader:
            state = state.to(torch.float32)
            policy = policy.to(torch.float32)
            value = value.to(torch.float32)

            state = state.to(net.device) if hasattr(net, 'device') else state
            policy = policy.to(state.device)
            value = value.to(state.device)

            pred_policy, pred_value = net(state)

            loss_policy = -torch.sum(policy * F.log_softmax(pred_policy, dim=1)) / policy.size(0)
            loss_value = loss_fn_value(pred_value.view(-1, 1), value)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}")


def evaluate_models(new_net, old_net, num_games=20):
    wins = 0
    for i in range(num_games):
        player1 = MCTS(new_net, num_simulations=50)
        player2 = MCTS(old_net, num_simulations=50)

        game = Gomoku()
        current = player1 if i % 2 == 0 else player2
        while not game.is_terminal():
            _, action = current.run(game)
            game = game.apply_action(action)
            current = player2 if current == player1 else player1

        winner = game.get_winner()
        if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
            wins += 1

    return wins / num_games


if __name__ == "__main__":
    net = AlphaZeroNet(board_size=BOARD_SIZE)
    mcts = MCTS(net, num_simulations=50)

    buffer = ReplayBuffer(max_size=10000)

    for episode in range(10):  # try 10 games to start
        print(f"Self-play game {episode+1}")
        game_data = play_self_play_game(mcts, board_size=8)
        buffer.add(game_data)
        train_network(net, buffer.sample(batch_size=256), epochs=5)
        print("---------------------------")

    print("Training complete. Saving model...")

    

    # Save the model with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), f"models/gomoku_{timestamp}.pt")


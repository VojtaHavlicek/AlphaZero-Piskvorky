from typing import List
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

 # --- Parameters ---
BOARD_SIZE = 5
WIN_LENGTH = 4
NUM_EPISODES = 100
NUM_SELF_PLAY_GAMES = 100
BATCH_SIZE = 64
MODEL_DIR = "models"

class ReplayBuffer:
    def __init__(self, max_size=10_000):
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

BOARD_SIZE = 5  # Default board size for Gomoku
WIN_LENGTH = 4  # Default win length for Gomoku

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


# Parallel self-play helper
from multiprocessing import get_context

def _self_play_worker(args):
    state_dict, board_size, net_cls, mcts_params = args
    device = torch.device('cpu')
    net = net_cls(board_size=board_size).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    mcts = MCTS(net, **mcts_params)
    # Use the same Gomoku from games
    from games import Gomoku
    def play_game():
        game = Gomoku(size=board_size)
        history = []
        move_num = 0
        while not game.is_terminal():
            temp = 1.0 if move_num < 10 else 0.01
            pi, action = mcts.run(game, temperature=temp)
            history.append((game.encode(device=device).squeeze(0), pi.numpy(), game.current_player))
            game = game.apply_action(action)
            move_num += 1
        winner = game.get_winner()
        return [(s, p, 1 if winner==cp else -1 if winner==-cp else 0)
                for s,p,cp in history]
    return play_game()

class ParallelTrainer:
    def __init__(self, trainer, num_workers=4):
        self.trainer = trainer
        self.num_workers = num_workers

    def collect_parallel_self_play(self, num_games):
        state_dict = self.trainer.net.state_dict()
        board_size = self.trainer.game_class().size
        mcts_params = dict(c_puct=self.trainer.mcts_class.c_puct,
                            num_simulations=self.trainer.mcts_class.num_simulations)
        args = [(state_dict, board_size, type(self.trainer.net), mcts_params)] * num_games
        with get_context('spawn').Pool(self.num_workers) as pool:
            for game_data in pool.map(_self_play_worker, args):
                self.trainer.buffer.add(game_data)


from torch.utils.data import Dataset 

class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return state, policy, torch.tensor([value], dtype=torch.float32)
    


def train_network(net, examples, epochs=5, batch_size=64, lr=1e-3):
    dataset = AlphaZeroDataset(examples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    net.train()
    for epoch in range(epochs):
        total_loss = 0
        for state, policy, value in dataloader:
            device = next(net.parameters()).device
            state = state.to(device)
            policy = policy.to(device)
            value = value.to(device)

            pred_policy, pred_value = net(state)

            loss_policy = -torch.sum(policy * F.log_softmax(pred_policy, dim=1)) / policy.size(0)
            loss_value = loss_fn_value(pred_value.view(-1, 1), value)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}")


def evaluate_models(new_net, old_net, num_games=NUM_SELF_PLAY_GAMES):
    wins = 0
    for i in range(num_games):
        player1 = MCTS(new_net, num_simulations=50)
        player2 = MCTS(old_net, num_simulations=50)

        game = Gomoku(size=BOARD_SIZE, win_length=WIN_LENGTH)
        current = player1 if i % 2 == 0 else player2
        while not game.is_terminal():
            _, action = current.run(game)
            game = game.apply_action(action)
            current = player2 if current == player1 else player1

        winner = game.get_winner()
        if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
            wins += 1

    return wins / num_games


# === Model promoter ===
import os
from datetime import datetime
from torch.multiprocessing import cpu_count

class ModelPromoter:
    def __init__(self, model_dir, threshold=0.55):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.best_path = None
        self.threshold = threshold

    def promote_if_better(self, candidate_net, base_net, win_rate, metadata=None):
        if win_rate >= self.threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
            torch.save(candidate_net.state_dict(), model_path)
            self.best_path = model_path
            print(f"\n✅ New model promoted: {model_path} (win rate: {win_rate:.2%})")
            if metadata:
                print("Metadata:", metadata)
        else:
            print(f"\n❌ Candidate rejected (win rate: {win_rate:.2%})")

# === Parallel self-play helper ===
def self_play_worker(task_queue:'mp.Queue', result_queue:'mp.Queue', net_state_dict, board_size, win_length):
    import time
    from queue import Empty
    import torch.multiprocessing as mp
    from net import AlphaZeroNet  # import locally for multiprocessing compatibility
    torch.set_num_threads(1)

    # Recreate model and load weights
    model = AlphaZeroNet(board_size=board_size)
    model.load_state_dict(net_state_dict)
    model.eval()

    torch.set_num_threads(1) # Ensure single-threaded execution in worker
    mcts = MCTS(model)
    while True: 
        try:
            task_id = task_queue.get_nowait()
        except Empty: 
            break

        try:
            game = Gomoku(size=board_size, win_length=win_length)
            history = []
            move_num = 0
            start_time = time.time()

            while not game.is_terminal():
                temp = 1.0 if move_num < 10 else 0.01
                policy, action = mcts.run(game, temperature=temp)
                state = game.encode().squeeze(0)
                history.append((state, policy, game.current_player))
                game = game.apply_action(action)
                move_num += 1

            winner = game.get_winner()
            duration = time.time() - start_time
            print(f"[Worker {task_id}] Game {task_id} finished in {move_num} moves ({duration:.2f}s), winner: {winner}")

            data = [
                (state, policy, 1.0 if winner == player else -1.0 if winner == -player else 0.0)
                for state, policy, player in history
            ]
            result_queue.put(data)

            print(f"[Worker {task_id}] Game data added to result queue.")

        except Exception as e:
            print(f"[Worker {task_id}] Error:", e)
            break

    print(f"[Worker {task_id}]: Process terminated ")


# === Self-play manager ===
def run_self_play_parallel(net, num_games=10, num_workers=None, board_size=BOARD_SIZE, win_length=WIN_LENGTH):
    import torch.multiprocessing as mp
    from queue import Empty

    if num_workers is None:
        num_workers = min(cpu_count(), num_games)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    net_state_dict = net.to('cpu').state_dict()

    for i in range(num_games):
        task_queue.put(i)

    workers:List[mp.Process]= [
        mp.Process(target=self_play_worker, args=(task_queue, result_queue, net_state_dict, board_size, win_length))
        for _ in range(num_workers)
    ]

    print(f"[Self-Play] All workers started, waiting for results. Awaiting {num_workers} workers...")
    for w in workers:
        w.start()
    print("[Self-Play] Workers started, processing games...")
    for i, w in enumerate(workers):
        print(f"[Debug] Worker {i} alive: {w.is_alive()}")
    #for w in workers:
    #    w.join()
    #    print(f"[Self-Play] Worker {w.pid} joined.")

    print("[Self-Play] All workers joined, collecting results...")


    results = []
    for _ in range(num_games):
        try:
            game_data = result_queue.get(timeout=10)
            results.append(game_data)
        except Empty:
            print("[Self-Play] Timeout waiting for result.")
            break


    print("[Self-Play] All games completed, processing results...")


    # Summary log
    num_games = len(results)
    num_moves = sum(len(game) for game in results)
    avg_moves = num_moves / num_games if num_games else 0
    print(f"[Self-Play] Collected {num_games} games, average moves: {avg_moves:.1f}")
    return results



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = AlphaZeroNet(board_size=BOARD_SIZE).to(device)
    promoter = ModelPromoter(MODEL_DIR)
    buffer = ReplayBuffer()

    best_net = AlphaZeroNet(board_size=BOARD_SIZE).to(device)
    best_net.load_state_dict(net.state_dict())

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n=== Episode {episode} ===")

        print("SELF-PLAYING GAMES...")
        data = run_self_play_parallel(best_net, num_games=NUM_SELF_PLAY_GAMES,
                                      board_size=BOARD_SIZE, win_length=WIN_LENGTH)
        
        data = [x for game in data for x in game]  # Flatten list of games
        buffer.add(data)

        print(f"Buffer size: {len(buffer)}")


        # Train network on buffer
        print("TRAINING NETWORK...")
        train_network(net, list(buffer.sample(1024)), epochs=5)

        # Evaluate model against best
        win_rate = evaluate_models(net, best_net, num_games=NUM_SELF_PLAY_GAMES)

        # Attempt promotion
        promoter.promote_if_better(net, best_net, win_rate, metadata={"episode": episode, "buffer_size": len(buffer)})

        # If promoted, update best_net
        if promoter.best_path:
            best_net.load_state_dict(torch.load(promoter.best_path, map_location=device))

        # Ensure the net is on the correct device
        net = net.to(device)



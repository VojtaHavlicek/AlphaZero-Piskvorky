#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Self-play implementation for AlphaZero algorithm.
License: MIT
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from alphazero.replay_buffer import ReplayBuffer
from alphazero.promoter import ModelPromoter
from mcts import MCTS
from games import Gomoku
from net import GomokuNet


 # --- Parameters ---
BOARD_SIZE = 3
WIN_LENGTH = 3
NUM_EPISODES = 100
NUM_SELF_PLAY_GAMES = 100
BATCH_SIZE = 64
MODEL_DIR = "models"

def play_self_play_game(mcts:MCTS):
    game = Gomoku(board_size=BOARD_SIZE, win_length=WIN_LENGTH)
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
        game = Gomoku(board_size=board_size)
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
    


def train_network(net, examples, epochs=10, batch_size=64, lr=1e-3):
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

from tqdm import tqdm

def evaluate_models(new_net, old_net, num_games=NUM_SELF_PLAY_GAMES):
    wins = 0
    for i in tqdm(range(num_games)):
        player1 = MCTS(new_net, num_simulations=50)
        player2 = MCTS(old_net, num_simulations=50)

        game = Gomoku(board_size=BOARD_SIZE, win_length=WIN_LENGTH)
        current = player1 if i % 2 == 0 else player2
        while not game.is_terminal():
            _, action = current.run(game)
            game = game.apply_action(action)
            current = player2 if current == player1 else player1

        winner = game.get_winner()
        if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
            wins += 1

        # terminate selfplay early.
        if i >= 9 and (wins/(i+1) > 0.60 or wins/(i+1) < 0.40):
            print("early termination")
            return wins/(i+1)

    return wins / num_games


# === Model promoter ===
from torch.multiprocessing import cpu_count

# === Parallel self-play helper ===
def self_play_worker(task_queue:'mp.Queue', result_queue:'mp.Queue', net_state_dict, board_size, win_length):
    import time
    from queue import Empty
    import torch.multiprocessing as mp
    from net import GomokuNet  # import locally for multiprocessing compatibility
    torch.set_num_threads(1)

    # Recreate model and load weights
    model = GomokuNet(board_size=board_size)
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
            game = Gomoku(board_size=board_size, win_length=win_length)
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




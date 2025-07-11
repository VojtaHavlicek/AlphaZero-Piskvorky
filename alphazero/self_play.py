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
from mcts import MCTS
from tqdm import tqdm


 # --- Parameters ---
NUM_EPISODES = 100
NUM_SELF_PLAY_GAMES = 100
BATCH_SIZE = 64
MODEL_DIR = "models"

import time
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import List, Type
from mcts import MCTS

import time
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import List, Type, Callable
from mcts import MCTS

def default_temperature_schedule(move: int) -> float:
    return 1.0 if move < 10 else 0.01

class SelfPlayManager:
    def __init__(
        self,
        net: torch.nn.Module,
        game_class: Type,
        mcts_params: dict = None,
        temperature_schedule: Callable[[int], float] = default_temperature_schedule,
    ):
        self.net = net
        self.net_class = type(net)
        self.game_class = game_class
        self.mcts_params = mcts_params or {"num_simulations": 50}
        self.temperature_schedule = temperature_schedule

    def _worker(self, task_queue: 'mp.Queue', result_queue: 'mp.Queue', state_dict):
        torch.set_num_threads(1)

        model = self.net_class()
        model.load_state_dict(state_dict)
        model.eval()

        mcts = MCTS(model, **self.mcts_params)

        while True:
            try:
                task_id = task_queue.get_nowait()
            except Empty:
                break

            try:
                game = self.game_class()  
                history = []
                move_num = 0
                start_time = time.time()

                while not game.is_terminal():
                    temp = self.temperature_schedule(move_num)
                    policy, action = mcts.run(game, temperature=temp)
                    state = game.encode().squeeze(0)
                    history.append((state, policy, game.current_player))
                    game = game.apply_action(action)
                    move_num += 1

                    #print(game)
                    #print("-------------")

                winner = game.get_winner()
                duration = time.time() - start_time
              
                

                #print(f"Encode: \n {game.encode()} \n -----------------")
                # Encode the terminal state and prepare the data


                state = game.encode().squeeze(0)  # Squeeze to remove batch dimension
                #print(f"Squeeze: \n {state} \n -----------------")

                # Check if the encoding works: 
                data = [
                    (state, policy, 1.0 if winner == p else -1.0 if winner == -p else 0.0)
                    for state, policy, p in history
                ] # This is the full selfplay history. 

                # print(f"Data: \n {data} \n -----------------")
                result_queue.put(data)
               

            except Exception as e:
                print(f"[Worker {task_id}] Error:", e)
                break

        # Terminated 


    def generate_self_play(self, num_games: int, num_workers: int = None) -> List:
        """
        Generate self-play games using multiple workers.

        Args:
            num_games (int): number of games to generate
            num_workers (int, optional): number of parallel workers. Defaults to min(cpu, num_games)

        Returns:
            List[List[Tuple[Tensor, Tensor, Tensor]]]: List of game histories
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), num_games)

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        state_dict = self.net.to("cpu").state_dict()

        for i in range(num_games):
            task_queue.put(i)

        workers = [
            mp.Process(target=self._worker, args=(task_queue, result_queue, state_dict))
            for _ in range(num_workers)
        ]

        for w in workers:
            w.start()

        print(f"[SelfPlayManager] Collecting {num_games} games with {num_workers} workers...")

        results = []
        with tqdm(total=num_games, desc="Self-play", ncols=80) as pbar:
            for _ in range(num_games):
                try:
                    data = result_queue.get(timeout=60)
                    results.append(data)
                    pbar.update(1)
                except Empty:
                    print("[SelfPlayManager] Timeout while waiting for result.")
                    break

        for w in workers:
            w.join()

        print(f"[SelfPlayManager] Collected {len(results)} games.")
        return results

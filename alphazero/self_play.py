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
from monte_carlo_tree_search import MCTS
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
from monte_carlo_tree_search import MCTS

import time
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import List, Type, Callable
from monte_carlo_tree_search import MCTS

def default_temperature_schedule(move: int) -> float:
    if move < 3:
        # Early game, high temperature
        return 1.0
    if move < 6:
        # Mid game, moderate temperature
        return 0.1
    
    # Late game, low temperature
    return 0.01

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
        self.mcts_params = mcts_params or {"num_simulations": 200}
        self.temperature_schedule = temperature_schedule


    def _worker(self, task_queue: 'mp.Queue', result_queue: 'mp.Queue', state_dict):

        import numpy as np
        torch.set_num_threads(1)

        policy_value_net = self.net_class()
        policy_value_net.load_state_dict(state_dict)
        policy_value_net.eval()

        mcts = MCTS(
            game_class=self.game_class,
            policy_value_net=policy_value_net)

        while True:
            try:
                task_id = task_queue.get_nowait()
            except Empty:
                break

            try: 
                game_state = self.game_class()
                history = []
                move_num = 0

                while not game_state.is_terminal():
                    temp = self.temperature_schedule(move_num)
                    #print(f"[Worker {task_id}] Starting move {move_num} with temperature {temp:.2f}")
                    policy, action = mcts.run(game_state=game_state, 
                                              temperature=temp, 
                                              add_exploration_noise=True)
                    
                    
                    #print(f"[Worker {task_id}] Policy: {policy.cpu().numpy().round(3)}")
                    state = game_state.encode().squeeze(0)
                    history.append((state, policy, game_state.current_player))
                    game_state = game_state.apply_action(action)
                    move_num += 1
                    #print(f"[Worker {task_id}] Move {move_num}: Player {game.current_player}, Action: {action}, Temp: {temp:.2f} \n{game}")
                #print(f"[Worker {task_id}] Game finished after {move_num} moves.")
                winner = game_state.get_winner()
                state = game_state.encode().squeeze(0)  

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


    def generate_self_play(self, 
                           num_games: int, 
                           num_workers: int = None, 
                           flatten=True) -> List:
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
        with tqdm(total=num_games, desc="[SelfPlayManager] Self-play", ncols=80) as pbar:
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

        # NOTE: Important - 
        # Flatten the results so this is plug an 
        if flatten:
            return [sample for game in results for sample in game]
        return results
       

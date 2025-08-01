#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Self-play implementation for AlphaZero algorithm.
License: MIT
"""

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from queue import Empty
from collections.abc import Callable
from controller import NeuralNetworkController, make_policy_value_fn
from mcts import MCTS
from games import X, O, DRAW, Gomoku
from net import GomokuNet  # Assuming GomokuNet is defined in net.py

def default_temperature_schedule(move: int) -> float:
    if move < 3:
        return 1.0
    if move < 6:
        return 0.1
    return 0.01

def _worker(task_queue: "mp.Queue", result_queue: "mp.Queue",
            state_dict, mcts_params, temperature_schedule):
    """Worker that generates self-play games on CPU."""
    torch.set_num_threads(1)

    # Build fresh network and controller on CPU
    controller = NeuralNetworkController(GomokuNet(device="cpu"), device="cpu")
    controller.net.load_state_dict(state_dict)
    controller.net.eval()

    mcts = MCTS(policy_value_fn=make_policy_value_fn(controller), **mcts_params)

    while True:
        try:
            task_id = task_queue.get_nowait()
        except Empty:
            break

        try:
            game_state = Gomoku()
            history = []
            move_num = 0

            while not game_state.is_terminal():
                temp = temperature_schedule(move_num)
                policy, action = mcts.run(
                    root_state=game_state,
                    temperature=temp,
                    add_root_noise=True
                )
                if game_state.current_player not in (X, O):
                    raise ValueError(f"Invalid current player: {game_state.current_player}")

                history.append((game_state.encode("cpu"), policy, game_state.current_player))
                game_state = game_state.apply_action(action)
                move_num += 1

            game_result = game_state.get_game_result()
            if game_result not in (X, O, DRAW):
                raise ValueError(f"Invalid game result: {game_result}")

            data = [(state, policy, 0 if game_result == DRAW else 1 if player == game_result else -1)
                    for (state, policy, player) in history]
            result_queue.put(data)

        except Exception as e:
            print(f"[Worker {task_id}] Error:", e)
            break


class SelfPlayManager:
    def __init__(
        self,
        controller: NeuralNetworkController,
        device, 
        mcts_params: dict = None,
        temperature_schedule: Callable[[int], float] = default_temperature_schedule,

    ):
        self.controller = controller
        self.mcts_params = mcts_params or {"num_simulations": 100}
        self.temperature_schedule = temperature_schedule
        self.device = device

    def _augment_symmetries(self, state_tensor, policy_tensor):
        """
        Generate rotation-based symmetries of (state, policy).
        state_tensor: shape (1, C, H, W)
        policy_tensor: shape (H*W,)
        """
        # TODO: Not implemented yet.
        H = state_tensor.size(2)
        W = state_tensor.size(3)
        policy_grid = policy_tensor.view(H, W)

        symmetries = []
        for k in range(4):
            rot_state = torch.rot90(state_tensor, k, [2, 3])
            rot_policy = torch.rot90(policy_grid, k, [0, 1]).flatten()
            symmetries.append((rot_state.clone(), rot_policy.clone()))
        return symmetries

    def generate_self_play(self, num_games: int, num_workers: int = None, flatten=True) -> list:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), num_games)

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        for i in range(num_games):
            task_queue.put(i)

        # Ensure model weights are on CPU
        state_dict = {k: v.cpu() for k, v in self.controller.net.state_dict().items()}

        # Spawn workers
        workers = [
            mp.Process(
                target=_worker,
                args=(task_queue, result_queue, state_dict,
                      self.mcts_params, self.temperature_schedule)
            )
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
                    print("[SelfPlayManager] Timeout waiting for result.")
                    break

        for w in workers:
            w.join()

        print(f"[SelfPlayManager] Collected {len(results)} games.")
        return [sample for game in results for sample in game] if flatten else results

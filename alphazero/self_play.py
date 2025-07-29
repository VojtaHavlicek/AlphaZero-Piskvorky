#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Self-play implementation for AlphaZero algorithm.
License: MIT
"""

import torch
from mcts import MCTS
from tqdm import tqdm
from games import Game, Gomoku, X, O, DRAW
from collections.abc import Callable
from queue import Empty
import torch.multiprocessing as mp


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
        game_class: type,
        mcts_params: dict = None,
        temperature_schedule: Callable[[int], float] = default_temperature_schedule,
    ):
        # The neural network model. 
        # It contains the policy and value head; policy head is used for MCTS search and is learned from MCTS 
        # simulations, value head is used to predict the game outcome.
        self.net = net 
        self.net_class = type(net)
        self.game_class = game_class
        self.mcts_params = mcts_params or {"num_simulations": 100}
        self.temperature_schedule = temperature_schedule

    def _worker(self, task_queue: "mp.Queue", result_queue: "mp.Queue", state_dict):
        torch.set_num_threads(1)

        policy_value_net = self.net_class()
        policy_value_net.load_state_dict(state_dict)
        policy_value_net.eval()

        mcts = MCTS(game_class=self.game_class, net=policy_value_net, **self.mcts_params)

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


                    policy, action = mcts.run(
                        game_state=game_state,
                        temperature=temp,
                        add_exploration_noise=True,
                    )

                    print(f"[Trainer]: policy {policy}")
                    


                    state = game_state
                    if game_state.current_player not in (X, O):
                        raise ValueError(
                            f"Invalid current player: {game_state.current_player}. Must be 'X' or 'O'."
                        )
                    history.append((state, policy, game_state.current_player))
                    game_state = game_state.apply_action(action)
                   

                winner = game_state.get_winner()

                data = [] 
                for state, policy, current_player in history: 
                    value = 0 if winner == DRAW else 1 if current_player == winner else -1

                    for _ in range():
                        # Rotate the state and policy
                        state = state.rot90()
                        policy = policy.rot90()

                        data.append(
                            (
                                state,  # Rotated game state
                                policy,  # Rotated policy vector
                                value,  # Value
                            )
                        )
               

                # print(f"Data: \n {data} \n -----------------")
                result_queue.put(data)

            except Exception as e:
                print(f"[Worker {task_id}] Error:", e)
                break

        # Terminated

    def get_symmetric_data(self, play_data):
        """
        Args: 
            Play data: [(state, policy, winner), ...]
        """

        symmetric_data = []
        for state, policy, winner in play_data:
            # Add the original data
            symmetric_data.append((state, policy, winner))

            # Create a symmetric version of the state
            for i in range(4):
                state = state.rot90()
                policy = policy.rot90()

                symmetric_data.append((state, policy, winner))

    def generate_self_play(
        self, 
        num_games: int, 
        num_workers: int = None, 
        flatten=True
    ) -> list:
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

        print(
            f"[SelfPlayManager] Collecting {num_games} games with {num_workers} workers."
        )

        results = []
        with tqdm(
            total=num_games, 
            desc="[SelfPlayManager] Self-play", 
            ncols=80
        ) as pbar:
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

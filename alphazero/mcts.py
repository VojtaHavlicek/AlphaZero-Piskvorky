#!/usr/bin/env python3
"""
Filename: promoter.py
Author: Vojtěch Havlíček
Created: 2025-07-11
Description: Monte Carlo Tree Search (MCTS) implementation for the AlphaZero algorithm.
License: MIT
"""
import random
import numpy as np
import torch
import torch.nn
from games import Gomoku
from games import X, O, DRAW

DEFAULT_CACHE_SIZE = 500_000  # Default size for the evaluation cache
DEFAULT_NUM_SIMULATIONS = 10_000  # Default number of simulations per move
DEFAULT_EXPLORATION_STRENGTH = 5  # Default exploration strength for UCB


class Node:
    """
    Represents a node in the MCTS tree.
    """

    def __init__(self, 
                 state: Gomoku, 
                 parent=None, 
                 prior=1.0):
        
        self.parent = parent
        self.state = state
        self.prior = prior  # NOTE: This is the prior probability of choosing this node in the UCB formula
        self.children = {}  # Action -> Node mapping 

        self.N = 0  # Visit count
        self.W = 0.0  # Upper confidence bound (UCB) value
        self.Q = 0.0  # Q value estimate (mean value)

    def is_expanded(self) -> bool:
        return bool(self.children)
    

    def expand(self, policy_dict):
        """
        Expands the node by creating child nodes for each legal action.
        This method uses the provided policy priors to set the prior probabilities for each child node.
        This is the prior probability of choosing this node in the UCB formula.

        Args:
            policy_priors: List[float]: A list of prior probabilities for each legal action.
        """
        for action, probability in policy_dict.items(): 
            if action not in self.children:
                self.children[action] = Node(
                    game_state=self.state.apply_action(action),
                    parent=self,
                    prior=probability,
                )
    
    def select(self, exploration_strength):
        """
        Selects a child node based on the UCB score.

        Args:
            c_puct: float: Exploration strength parameter for UCB.
        """
        return max(self.children.items(), 
                   key=lambda action_node: action_node[1].get_value(exploration_strength))
    

    def update(self, value):
        """
        Updates the node with a new value.

        Args:
            value: float: The value to update the node with.
        """
        self.N += 1
        self.Q += 1.0*(value - self.Q) / self.N  # Incremental mean update


    def update_recursive(self, value):
        """
        Updates the node and propagates the value up to the parent nodes.

        Args:
            value: float: The value to update the node with.
        """
        if self.parent is not None:
            self.parent.update_recursive(-value)
        self.update(value)
    
    
    def get_value(self, exploration_strength):
        """
        Computes the UCB value for this node.

        Args:
            c_puct: (float) Exploration strength parameter for UCB.

        Returns:
            float: The UCB value for this node.
        """
        self.W = (exploration_strength * self.prior * (self.N**0.5)) / (1 + self.N)
        return self.Q + self.W
    
    def is_leaf(self): 
        return self.children == {}
    
    def is_root(self):
        return self.parent is None


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
    

class MCTS:
    def __init__(
        self,
        game_class: Gomoku,
        policy_value_fn, # Function to get policy and value. policy is a 2D array of (board_size, board_size) shape, value is a scalar.
        exploration_strength=DEFAULT_EXPLORATION_STRENGTH,
        num_simulations=DEFAULT_NUM_SIMULATIONS,
        cache_size=DEFAULT_CACHE_SIZE
    ):
        self.game_class = game_class  # Type of the game, e.g., TicTacToe
        self.policy_value_fn = policy_value_fn  # Function to get policy and value. Usually a neural network.
        self.exploration_strength = (
            exploration_strength  # Strength of exploration in UCB formula
        )
        self.num_playouts = num_simulations  # Number of simulations to run per move
        self.cache_size = cache_size  # Maximum size of the evaluation cache

        self.evaluation_cache = self._init_cache()
        print(f"[MCTS] exploration strength: {exploration_strength}, num_simulations: {num_simulations}")

        self.root = Node(None, 1.0) # Initialize root node with a dummy state and prior of 1.0


    def _get_policy_dict(self, policy):
        """
        Converts a policy (2D array) into a dictionary mapping actions to probabilities.

        Args:
            policy (_type_): _description_

        Returns:
            _type_: _description_
        """
        policy_dict = {}
        for row in len(policy):
            for col in len(policy[row]):
                action = (row, col)
                policy_dict[action] = policy[row][col]
        return policy_dict


    def _playout(self, state:Gomoku):
        """
        Runs a playout from the given game state to a terminal state.
        This method is used to simulate the outcome of a game from a given state.

        Args:
            state (Game): The current game state to simulate from.

        Returns:
            float: The value of the terminal state from the perspective of the current player.
        """
        node = self.root 

        while not node.is_leaf():
            action, node = node.select(self.exploration_strength)
            state = state.apply_action(action)

        policy, leaf_value = self.policy_value_fn(state) # Eval policy and value from the supplied policy-value function
        policy_dict = self._get_policy_dict(policy) # Convert policy to dictionary

        game_result = state.get_game_result()

        if game_result is None:
            node.expand(policy_dict)
        else:
            if game_result == DRAW:
                leaf_value = 0.0
            elif game_result == state.current_player:
                leaf_value = 1.0
            else:
                leaf_value = -1.0
          
        node.update_recursive(-leaf_value)  # Backpropagate the value up the tree


    def get_move_probabilities(self, 
                               state:Gomoku, 
                               temperature=1e-3):
        for _ in range(self.num_playouts):
            state = state.clone()  # Clone the state to avoid modifying the original
            self._playout(state)

        # NOTE: Why is this computed from visits? 

        actions_and_visits = [(action, node.N) for action, node in self.root.children.items()]
        actions, visits = zip(*actions_and_visits, strict=True)  # Unzip actions and visits
        action_probabilities = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))  # Add small value to avoid log(0)

        return actions, action_probabilities
    

    def update_with_move(self, 
                         last_move):
        """
        Moves down the tree to the node corresponding to the last move made in the game.

        Args:
            last_move (_type_): _description_
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None  # Reset parent to None for the new root
        else:
            self.root = Node(None, 1.0)

class MCTSPlayer:
    def __init__(self, 
                 game_class,
                 policy_value_fn,
                 exploration_strength=DEFAULT_EXPLORATION_STRENGTH,
                 num_simulations=DEFAULT_NUM_SIMULATIONS,
                 is_selfplay=False): 
        
        self.mcts = MCTS(
            game_class=game_class,
            policy_value_fn=policy_value_fn,
            exploration_strength=exploration_strength,
            num_simulations=num_simulations
        )

        self.is_selfplay = is_selfplay


    def get_action(self,
                   state:Gomoku, 
                   temperature=1e-3):
        """
        Get the action to take in the current state using MCTS.

        TODO: add an option to return the action probabilities.
        Args:
            state (Game): The current game state.
            temperature (float): Temperature parameter for action selection.

        Returns:
            tuple: The selected action and its probability.
        """
        actions, action_probabilities = self.mcts.get_move_probabilities(state, temperature)
        
        if self.is_selfplay:
            move = np.random.choice(actions, p=0.75*action_probabilities + 0.25*np.random.dirichlet(np.ones(len(actions))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(actions, p=action_probabilities)
            self.mcts.update_with_move(-1)  # Reset the root node after each move in non-selfplay mode

        return move



    


            


    
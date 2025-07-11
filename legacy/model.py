# ---- MODEL & MCTS ----
from dataclasses import dataclass
import mlx.nn as nn
import mlx.core as mx
import math
import random
from game import GameState
import numpy as np


class Node:
    """
    MCTS node representing a game state.
    """
    def __init__(self, state, parent=None, action=None):
        self.visits = 0
        self.total_value = 0.0
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}

        self.state = state.clone()
       

    def value(self) -> float:
        """
        Returns the value of the node.
        The value is the total value divided by the number of visits.
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    
    def is_fully_expanded(self) -> bool:
        """
        Checks if all actions from the current state have been explored.

        Returns:
            bool: True if all actions have been explored, False otherwise.
        """
        return len(self.children) == len(self.state.actions())

    def best_child(self, c=1.0) -> tuple:
        """
        Chooses the best child node based on UCT (Upper Confidence Bound for Trees). 
        NOTE: What's the theory behind UCT?

        Args:
            c (float, optional): Exploitation/Exploration coeff. Defaults to 1.0.

        Returns:
            tuple: (action, Node)
            
        """

        
        best_value = -math.inf
        best_action = None
        best_node = None
        for action, child in self.children.items():
            exploit = child.total_value / (child.visits + 1e-6)
            explore = c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + explore
            if score > best_value:
                best_value = score
                best_action = action
                best_node = child
        return best_action, best_node

    def expand(self):
        """
        Expands the node by adding a new child for one of the unexplored actions.

        Returns:
            Node: The newly created child node for the next action.
        If all actions have been explored, returns None.
        """
        for action in self.state.actions():
            if action not in self.children:
                next_state = self.state.clone()
                next_state.step(action)
                self.children[action] = Node(next_state, parent=self, action=action)
                return self.children[action]
        return None

    def backpropagate(self, value) -> None:
        """
        Backpropagates the value up the tree, updating the visits and total value. 
        Mutates the node and its parent nodes.

        Args:
            value (float): The value to backpropagate.
        """
        self.visits += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


    def get_policy_from_visits(root, temperature=1.0):
        """
        Computes the policy vector from the visit counts of the root node's children.

        Args:
            root (_type_): _description_
            temperature (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        visits = [child.visits for child in root.children.values()]
        total = sum(visits)
        policy = [visit / total for visit in visits]
        return mx.array(policy)


def mcts(game:GameState, model:nn.Module, num_simulations=50):
    """
    Runs a monte-carlo tree search for a given game.

    Args:
        game (GameState): game and game rules to simulate
        model (nn.Module): neural net model to use 
        num_simulations (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    root = Node(game)
    for _ in range(num_simulations):
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.state.is_terminal():
            _, node = node.best_child()
            if node is None:
                break

        # 2. Expansion
        if node is not None and not node.state.is_terminal():
            node = node.expand()

        # 3. Evaluation with a neural net
        if node is None:
            continue

        # 4. Rollout or neural network evaluation
        if model is not None:
            # Use the neural network to get policy and value
            state_input = node.state.encode().reshape(1, -1)  # shape: (1, 64)
            policy_logits, value = model(state_input)
            value = float(value.item())  # scalar
        else:
            value = rollout(node.state)
        
        node.backpropagate(value)


    action_visits = [(action, child.visits) for action, child in root.children.items()]
    best_action = max(action_visits, key=lambda x: x[1])[0]
    return best_action


def rollout(state, max_depth=100):
    """Perform a random rollout from the given state."""
    current_state = state.clone()
    for _ in range(max_depth):
        if current_state.is_terminal():
            return current_state.get_reward(current_state.get_current_player())
        actions = current_state.actions()
        if not actions:
            return 0.0
        action = random.choice(actions)
        current_state.step(action)
    return 0.0



class AZNet(nn.Module):
    """AlphaZero-style neural network for policy and value estimation."""
    def __init__(self, input_dim=9, hidden_dim=64, num_actions=9):
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x)) # first fully connected layer
        x = nn.relu(self.fc2(x)) # second fully connected layer
        policy = self.policy_head(x) # policy logits
        value = nn.tanh(self.value_head(x)) # value output
        return policy, value

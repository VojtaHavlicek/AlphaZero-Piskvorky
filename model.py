import mlx.core as mx
import mlx.nn as nn
import math
import random
from collections import defaultdict

# ---- MCTS ---- 
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state.clone()  # Deep copy of the state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.actions())
    
    def best_child(self, c=1.0):
        """
        Selects the child with the highest UCT value.
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
    

    def expand(self, model = None):
        actions = self.state.actions()
        for action in actions:
            if action not in self.children:
                next_state = self.state.clone()
                next_state.step(action)
                self.children[action] = Node(next_state, parent=self, action=action)
                return self.children[action]
            
    def backpropagate(self, value):
        """
        Backpropagates the value up the tree.
        """
        self.visits += 1
        self.total_value += value
        
        if self.parent is not None:
            self.parent.backpropagate(-value)


def mcts(game, model, num_simulations=1000):
    """
    Monte Carlo Tree Search (MCTS) algorithm.
    """
    root = Node(game)

    for _ in range(num_simulations):
        node = root
        
        # Selection
        while node.is_fully_expanded() and not node.state.is_terminal():
            _, node = node.best_child()
        
        # Expansion
        if not node.state.is_terminal:
            node = node.expand()
        
        # Simulation
        sim_state = node.state.clone()
        value = rollout(sim_state)
        
        # Backpropagation
        node.backpropagate(value)
    
    action_visits = [(action, child.visits) for action, child in root.children.items()]
    best_action = max(action_visits, key=lambda x: x[1])[0]
    return best_action


def rollout(game, max_depth=100):
    for _ in range(max_depth):
        if game.is_terminal:
            return game.get_reward(game.get_current_player())
        actions = game.actions()
        action = random.choice(actions)
        game.step(action)
    return 0  # draw if not terminal within depth




# ---- MODEL LIBRARY ----

class AZNet(nn.Module): 
    """ 
    AlphaZero Network (AZNet) for policy and value estimation.
    """
    def __init__(self, input_dim=9, hidden_dim=64, num_actions=9):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Policy head? Value head? 

    def __call__(self, x):
        x = mx.relu(self.fc1(x))
        x = mx.relu(self.fc2(x))

        policy = self.policy_head(x)
        value = mx.tanh(self.value_head(x))

        return policy, value
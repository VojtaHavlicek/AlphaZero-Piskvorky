import numpy as np
import random
import math 
import mlx.core as mlx
from engine import GameState
from collections import defaultdict

class Node: 
    """ 
    MCTS node for representing the state of the game.
    """

    def __init__(self, 
                 state:GameState, 
                 parent=None):
        self.visits = 0
        self.value = 0.0
        self.state = state.clone()  # Clone the state to avoid modifying the original
        self.parent = parent
        self.children = []
        self.is_expanded = False

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded.
        A node is fully expanded if all its children have been created.
        """
        return len(self.children) == len(list(self.state.actions()))


    def best_child(self, c=1.4):
        """
        Select the best child node based on UCT (Upper Confidence Bound for Trees).
        """
        best_score = float('-inf')
        best_action = None
        best_node = None

        for action, child in self.children.items():
            exploit = child.total_value / (child.visits + 1e-6)  # Avoid division by zero
            explore = c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + explore

            if score > best_score:
                best_score = score 
                best_action = action
                best_node = child 

        return best_action, best_node
    
    def expand(self): 
        legal_actions = self.state.actions()
        for action in legal_actions:
            if action not in self.children: 
                new_state = self.state.clone() 
                new_state.step(action)
                self.children[action] = Node(new_state, parent=self, action=action)

                return self.children[action]
            

    
    def backpropagate(self, value):
        self.visits += 1
        self.total_value += value

        if self.parent: 
            self.parent.backpropagate(-value) # switch objectives


# --- Rollouts ---
def rollout(state, max_depth=100):
    for _ in range(max_depth):
        if state.is_terminal(): 
            return state.get_reward() 
        
        actions = state.actions() 
        action = random.choice(state)
        state.step(action) 
    return 0 # terminal not within depth 

# --- Engine --- 
def mcts_search(root, num_simulations=100):
    root = Node(root)

    for _ in range(num_simulations):
        node = root

        # 1. Node selection: 
        while node.is_fully_expanded() and not node.state.is_terminal(): 
            _, node = node.best_child() 

        # 2. Expand if not terminal.
        if not node.is_termina(): 
            node = node.expand()

        # 3. simulation
        sim_game = node.state.clone()
        value = rollout(sim_game) # <---- Implement rollouts? 

        # 4. Backpropagation
        node.backpropagate(value)

    
    # Choose most visited action
    action_visits = [(action, child.visits) for action, child in root.children.items()]
    best_action = max(action_visits, key = lambda x: x[1])[0]
    return best_action


        
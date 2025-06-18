import torch
from games import Gomoku

class Node: 
    def __init__(self, game:Gomoku, parent=None, prior=0.0):
        self.game = game 
        self.parent = parent
        self.prior = prior 

        self.children = {}

        self.N = 0 # Number of visits
        self.W = 0.0 # Total value of the node
        self.Q = 0.0 # Average value of the node

    def is_expanded(self) -> bool:
        return len(self.children) > 0
    

def ucb_score(parent, child, c_puct):
    """Calculate the UCB score for a child node."""
    return child.Q + c_puct * child.prior * (parent.N ** 0.5) / (1 + child.N)
    



class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100):
        self.net = net 
        self.device = next(self.net.parameters()).device
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, game, temperature) -> tuple: 
        """ Run MCTS on the given game state.
        Args:
            game (Gomoku): The game state to run MCTS on.
        Returns:
            tuple: A tuple containing the policy and the best action.
        """
        root = Node(game)

        for _ in range(self.num_simulations):
            node = root
            path = [node]

            while node.is_expanded() and not node.game.is_terminal():
                action, node = self.select_child(node)
                path.append(node) 

            if not node.game.is_terminal():
                policy, value = self.evaluate(node.game)
                self.expand(node, policy)
            else: 
                value = node.game.get_winner()

            self.backpropagate(path, value if node.game.current_player == -1 else -value)

        return self.get_policy(root, temperature)
    
    def evaluate(self, game):
        """Evaluate the game state using the neural network."""
        with torch.no_grad():
            encoded = game.encode()
            encoded = encoded.to(next(self.net.parameters()).device)
            policy_logits, value = self.net(encoded)
            policy = torch.softmax(policy_logits, dim=-1)
        return policy.squeeze(0).cpu().numpy(), value.item() 
    

    def expand(self, node, policy):
        legal_actions = node.game.get_legal_actions()
        for action in legal_actions:
            index = action[0] * node.game.size + action[1]
            node.children[action] = Node(
                game=node.game.apply_action(action),
                parent=node,
                prior=policy[index]
            )

    def select_child(self, node):
        return max(
            node.children.items(),
            key=lambda item: ucb_score(node, item[1], self.c_puct)
        )
    
    def backpropagate(self, path, value):
        """Backpropagate the value through the path."""
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    def get_policy(self, root, temperature=1.0):
        """Get the policy from the root node."""
        visit_counts = torch.tensor([
            child.N for child in root.children.values()
        ], dtype=torch.float32)

        actions = list(root.children.keys())

        if temperature == 0:
            best_action = actions[visit_counts.argmax().item()]
            policy = torch.zeros(root.game.size * root.game.size)
            index = best_action[0] * root.game.size + best_action[1]
            policy[index] = 1.0
            return policy, best_action
        
        if visit_counts.sum() == 0:
            # All children unexplored (rare, but safe fallback)
            probs = torch.ones(len(actions)) / len(actions)
        else:
            counts = visit_counts ** (1 / temperature)
            probs = counts / counts.sum()

        # Handle NaN, inf, and negative values in probabilities
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0)
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / len(probs)
        else:
            probs /= probs.sum()

        policy = torch.zeros(root.game.size * root.game.size)
        for i, action in enumerate(actions):
            index = action[0] * root.game.size + action[1]
            policy[index] = probs[i]

        action = actions[torch.multinomial(probs, 1).item()]
        return policy, action
        
if __name__ == "__main__":
    from net import AlphaZeroNet  # Assuming you have a neural network defined in net.py
    # Example usage
    game = Gomoku(size=8)
    net =  AlphaZeroNet(board_size=8) # Replace with your neural network instance
    mcts = MCTS(net, num_simulations=50)

    policy, action = mcts.run(game)
    print("Policy:", policy)
    print("Best action:", action)

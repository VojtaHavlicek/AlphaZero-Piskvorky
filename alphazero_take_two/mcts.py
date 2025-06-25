import torch
from games import Gomoku

class Node:
    def __init__(self, game: Gomoku, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)


def ucb_score(parent: Node, child: Node, c_puct: float) -> float:
    return child.Q + c_puct * child.prior * (parent.N ** 0.5) / (1 + child.N)


class MCTS:
    def __init__(self, net, c_puct=1.0, num_simulations=100, cache_size=100_000):
        self.net = net
        self.device = next(net.parameters()).device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
        # LRU evaluation cache (Least Recently Used)
        from collections import OrderedDict 
        self.eval_cache = OrderedDict()
        self.cache_size = cache_size


    def _game_to_key(self, game:Gomoku):
        return tuple(game.encode(device='cpu').squeeze(0).flatten().tolist())


    def run(self, game: Gomoku, temperature: float = 1.0):
        root = Node(game)
        pending = []  # list of (node, path)

        # 1. Simulations: collect leaf nodes
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.is_expanded() and not node.game.is_terminal():
                action, node = max(
                    node.children.items(),
                    key=lambda item: ucb_score(node, item[1], self.c_puct)
                )
                path.append(node)
            if node.game.is_terminal():
                value = node.game.get_winner()
                self._backpropagate(path, value)
            else:
                pending.append((node, path))

        # 2. Batch evaluate pending leaves
        if pending:
            states = []
            idxs = []

            for i, (node, _) in enumerate(pending):
                key = self._game_to_key(node.game)
                if key in self.eval_cache:
                    continue
                states.append(node.game.encode(device=self.device))
                idxs.append(key)

            if states:
                batch = torch.cat(states, dim=0)
                with torch.no_grad():
                    logits, values = self.net(batch)
                    probs = torch.softmax(logits, dim=1)
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                # Cache results 
                for key, p, v in zip(idxs, probs, values):
                    self.eval_cache[key] = (p.cpu(), v.item())
                    # Maintain LRU size 
                    if len(self.eval_cache) > self.cache_size:
                        self.eval_cache.popitem(last=False)


            for node, path in pending: 
                key = self._game_to_key(node.game)
                policy_tensor, value = self.eval_cache[key]
                self._expand(node, policy_tensor.numpy())
                self._backpropagate(path, value)

        # 3. Compute final policy and action
        return self._get_policy(root, temperature)

    def _expand(self, node: Node, policy: list):
        for action in node.game.get_legal_actions():
            idx = action[0] * node.game.size + action[1]
            node.children[action] = Node(
                game=node.game.apply_action(action),
                parent=node,
                prior=policy[idx]
            )

    def _backpropagate(self, path: list, value: float):
        # value is from the perspective of the leaf's parent; flip each step
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    def _get_policy(self, root: Node, temperature: float):
        visits = torch.tensor([c.N for c in root.children.values()], device=self.device, dtype=torch.float32)
        actions = list(root.children.keys())
        if temperature == 0:
            best = actions[visits.argmax().item()]
            pi = torch.zeros(root.game.size**2, device=self.device)
            idx = best[0] * root.game.size + best[1]
            pi[idx] = 1.0
            return pi.cpu(), best
        if visits.sum() == 0:
            probs = torch.ones_like(visits) / len(visits)
        else:
            counts = visits ** (1.0 / temperature)
            probs = counts / counts.sum()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0)
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / len(probs)
        else:
            probs /= probs.sum()
        pi = torch.zeros(root.game.size**2, device=self.device)
        for a, p in zip(actions, probs):
            idx = a[0] * root.game.size + a[1]
            pi[idx] = p.item()
        action = actions[torch.multinomial(probs, 1).item()]
        return pi.cpu(), action

if __name__ == "__main__":
    from net import AlphaZeroNet
    game = Gomoku(size=8)
    net = AlphaZeroNet(board_size=8).to(torch.device('cpu'))
    mcts = MCTS(net, num_simulations=50)
    pi, act = mcts.run(game)
    print("Policy:", pi)
    print("Action:", act)


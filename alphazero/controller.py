import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, functional
import torch.nn.functional as functional
from tqdm import tqdm
from games import Gomoku  # Assuming Gomoku is defined in games.py


class AlphaZeroDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples  # list of (state, policy, value)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return (
            state.float(),
            policy,
            torch.tensor(value, dtype=torch.float32)
            if not torch.is_tensor(value)
            else value.float(),
        )
    
def make_policy_value_fn(controller):
    """
    Returns a callable policy_value_fn(state) -> (policy, value)
    that works on a single game state.
    """

    def policy_value_fn(state):
        # Encode single state into a tensor
        if isinstance(state, Gomoku):
            state_tensor = state.encode(controller.device).unsqueeze(0)  # [1, C, H, W]

        with torch.no_grad():
            policy_logits, value = controller.net(state_tensor)
            # Convert to 2D policy map
            board_size = state.board_size
            policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy().reshape(board_size, board_size)
            value = float(value.item())

        return policy, value

    return policy_value_fn


class NeuralNetworkController:
    """
    Use this for training and access to the model 
    """
    def __init__(self, 
                 net, 
                 device,
                 batch_size=2048):
        
        print(f"[Controller] Initializing NeuralNetworkController with device: {device}, batch_size: {batch_size}")
       
            
        self.device = device
        self.net = net.to(device).float()  # Ensure the model is in float32 format
        self.batch_size = batch_size
        self.training_history = []  # Store training history for analysis

        self.weight_decay = 1e-4  # Default weight decay for AdamW

        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(self.net.parameters(), 
                                           weight_decay=self.weight_decay) 

    
    def policy_value(self, state_batch): 
        """ 
        Input:
            state_batch: A batch of states (numpy array or torch tensor).
        Output:
            pred_policy: A numpy array of predicted policies for each state in the batch.
            pred_value: A numpy array of predicted values for each state in the batch.
        """
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.tensor(state_batch, dtype=torch.float32)
        
        state_batch = state_batch.to(self.device)

        with torch.no_grad():
            pred_policy_logits, pred_value = self.net(state_batch)
            pred_policy = torch.softmax(pred_policy_logits, dim=1)

        #with torch.no_grad():
        #           logits, value_tensor = self.net(state.encode(self.device).unsqueeze(0))
        #            policy = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy().reshape(
        #                state.board_size, 
        #                state.board_size
        #            )
        #            value = float(value_tensor.item())
        
        return pred_policy.cpu().numpy(), pred_value.cpu().numpy()


    def train_step(self, 
                   state_batch, 
                   mcts_probs, 
                   winner_batch, 
                   lr):
        """
        Perform a single training step.
        
        Args:
            state_batch: A batch of states (numpy array or torch tensor).
            mcts_probs: A batch of MCTS probabilities (numpy array or torch tensor).
            winner_batch: A batch of winners (numpy array or torch tensor).
            lr: Learning rate for the optimizer.
        """
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.tensor(state_batch, dtype=torch.float32)
        
        if not isinstance(mcts_probs, torch.Tensor):
            mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32)
        
        if not isinstance(winner_batch, torch.Tensor):
            winner_batch = torch.tensor(winner_batch, dtype=torch.float32)

        state_batch = state_batch.to(self.device).float()  # Ensure float32 format
        mcts_probs = mcts_probs.to(self.device).float()  # Ensure float32 format
        winner_batch = winner_batch.to(self.device).float()  # Ensure float32 format

        self.optimizer.zero_grad()
        self.optimizer.param_groups[0]['lr'] = lr

        pred_policy_logits, pred_value = self.net(state_batch)
        
        value_loss = functional.mse_loss(pred_value.view(-1), winner_batch)
        policy_loss = functional.cross_entropy(pred_policy_logits, mcts_probs.argmax(dim=1))
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.sum(mcts_probs * torch.log(pred_policy_logits + 1e-8)) / state_batch.size(0)
        return loss.item(), entropy.item()
   

    def train(self, examples, epochs=10):
        dataloader = self._prepare_data(examples)
        self.net.train()

        print(f"[Controller] Training started. Epochs: {epochs}, Batch size: {self.batch_size}, Learning rate: {self.optimizer.param_groups[0]['lr']}")

        for epoch in tqdm(range(epochs), desc="[Controller] Epochs", ncols=80):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0

            for state, policy, value in dataloader:
                state = state.to(self.device).float()  # Ensure float32 format
                policy = policy.to(self.device).float()  # Ensure float32 format
                value = value.to(self.device).float()  # Ensure float32 format

                # Perform a training step
                loss, entropy = self.train_step(state, policy, value, self.optimizer.param_groups[0]['lr'])
                
                total_loss += loss
                total_policy_loss += functional.cross_entropy(
                    self.net(state)[0], policy.argmax(dim=1)
                ).item()
                total_value_loss += functional.mse_loss(
                    self.net(state)[1].view(-1), value
                ).item()

            # Average the losses over the batch
            total_loss /= len(dataloader)
            total_policy_loss /= len(dataloader)

            print(f"[Controller] Epoch {epoch + 1}/{epochs}. Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}")
            self.training_history.append(
                {
                    "loss": total_loss,
                    "policy": total_policy_loss,
                    "value": total_value_loss,
                }
            )

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: Total={total_loss:.4f}, Policy={total_policy_loss:.4f}, Value={total_value_loss:.4f}"
                )

        print(
            f"[Controller] Training finished. Loss: {total_loss:.4f}, Policy Loss: {total_policy_loss:.4f}, Value Loss: {total_value_loss:.4f}"
        )


    def _prepare_data(self, examples):
        dataset = AlphaZeroDataset(examples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net = self.net.to(self.device).float()  # Ensure the model is in float32 format

    def eval(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

   
# AlphaZero - Piskvorky
This package implements a variant of AlphaZero for training AI models for perfect knowledge games. 
You can use this to train your own [Piskvorky](https://cs.wikipedia.org/wiki/Pi%C5%A1kvorky) playing model from self-play! 

## Quick Setup
### Train 5x5 - 4 in a row

1. Burn in: 

constants.py: 
```
BOARD_SIZE = 5, 
WIN_LENGTH = 4
NUM_EPISODES = 20
LEARNING_RATE = 1e-3
```

python train.py

This takes about 30 minutes on my laptop. 
It populates the example buffer and creates a baseline model.

2. Train: 
```
BOARD_SIZE = 5, 
WIN_LENGTH = 4
NUM_EPISODES = 100
LEARNING_RATE = 1e-4
```

python train.py

This takes a few hours on my laptop. You should see the AI gradually improving. 

3. Playtest
   python main.py

   If the AI doesn't play well, try to adjust parameters in consts and train a few more batches of episodes (you can increase number of selfplay simulations or play with learning rate). 



## Key Features
- AlphaZero reinforcement learning loop
- Custom neural net definition.
- Replay Buffer for self-play game data management.
- Model Promoter for self-play model management 

## Architecture

### Training Pipeline 
The training pipeline loops over three things: 

1. Self-play
2. Neural net training
3. Model eval and promotion 

#### Frequently Asked Questions: 
1. Self play games. 
    * How many games should I play?
        - For TicTacToe, choose 100-500 per iteration. 
        - For Gomoku or Othelo, 1_000-10_000.
    * What data to collect and how to structure things? 
        - The training examples should contain (state, pi, z), where state is the board state encoded as a tensor, pi is an MCTS-based policy (a probability distribution over legal moves at that state) and z is the final game result from the perspective of the player at that state (+1,-1, 0 for draw). Note that all game states contains the final result as a feature. All of the states in the history have to contain the final value. 
        - Replay Buffer can have about 10_000-100_000 samples here. 


2. Train neural net on the newly collected data, batch subsampled from the buffer
    * What batch size to choose and how many epochs? 
        - 64-128 for small boards like TicTacToe
        - 256+ on larger boards, GPU permitting. 
    * What neural network architecture to choose? 
        - board state is a 3x3xC tensor (C channels for current player, opponent, turn marker). (Understand better).
        - 2-3 convolutional layers (32 filters, 3x3 kernels, ReLU).
        - Flatten -> Fully connected layers.

        - Two heads: 
            - Policy head: dense -> softmax over actions (9 for 3x3)
            - Value head: dense -> scalar output (tanh activation for z in [-1,1])



3. Evaluate the neural net and promote the model if it did better. 
    * How many self play games to choose for evaluation? 
        - Use at least 50-100 games to compare the new model to the current.
        - If the new model wins more than 55-60%, promote it as the new best. 
        - This can be adaptive (using Elo ratings or statistical significance testing)

4. Value and Policy head
    * AlphaZero's neural net predicts two things for any given position:
        - Policy head: A probability distribution over all possible moves
        - Value head: Scalar that estimates the final outcome from a given position (win/loss/draw)

    * Policy head: predicts which moves are likely to be good 
        - used by MCTS as it does not explore moves uniformly
        - focuses on promising moves suggested by the network, speeding up search, improving quality
        - learns from **MCTS visit counts** during the training. 
    
    * Value head
        - predicts how good is the current position for **the player to move**! 
        - eliminates the need to rollout to terminal states during search. 
            - In classical MCTS before Zero, needed rollouts. Not here! 
            - Stop early and use the value estimate. 

        - Learns from **game outcomes** 

    


5. Miscelaneous notes:  
    * MCTS
        - Use temperature annealing for exploration early in training, low for precise eval later in the optimization 
        - Dirichlet noise: add to the root node during self-play to encourage exploration 
        - Very important: resolve policy search ties at random! 

    - Data augmentation: Augment data using rotation/reflections.



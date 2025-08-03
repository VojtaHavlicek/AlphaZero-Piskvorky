# AlphaZero - Piskvorky
An implementation of AlphaZero for training AI for perfect knowledge games. 
Use this to train your own [Piskvorky](https://en.wikipedia.org/wiki/Gomoku) model from self-play! 

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
        - For Gomoku, 1_000-10_000 new examples per self play.
        - The engine uses symmetry augmentation, so you need 8x less many games ~  100-150 selfplay games are just fine on 5x5 board. 
    * What data to collect and how to structure things? 
        - The training examples should contain (state, pi, z), where state is the board state encoded as a tensor, pi is an MCTS-based policy (a probability distribution over legal moves at that state) and z is the final game result from the perspective of the player at that state (+1,-1, 0 for draw). Note that all game states contains the final result as a feature. All of the states in the history have to contain the final value. 
        - Replay Buffer can have about 10_000-100_000 samples here. 


2. Train neural net on the newly collected data, batch subsampled from the buffer
    * What batch size to choose and how many epochs? 
        - 1024-2048 worked best. 
    * What neural network architecture to choose? 
        - Few convolutional layers in the stem, conv layers in the heads.
        - This will change - will add ResidualBlocks.

        - Two heads: 
            - Policy head: dense 
            - Value head: dense -> scalar output (tanh activation for z in [-1,1])



3. Evaluate the neural net and promote the model if it did better. 
    * How many self play games to choose for evaluation? 
        - Use at least 50 games to compare the new model to the current. The code uses symmetry augmentation by ~8x factor. 
        - If the new model wins more than 55%, promote it as the new best. 
        - This can be adaptive (TODO: using Elo ratings or statistical significance testing)

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


5. Miscelaneous notes:  
    * MCTS
        - Use temperature annealing for exploration early in training, low for precise eval later in the optimization 
        - Dirichlet noise: add to the root node during self-play to encourage exploration 
      
    - Data augmentation: Augment data using rotation/reflections.



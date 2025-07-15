# Strategy.ai
This package implements a suite of reinformcement learning algorithms for board game playing. 
It contains an AlphaZero implementation that you can use to train AI for games described in games.py. 
You can then play against the models. 

## Key Features
- Abstract class for turn based games. 
- AlphaZero like reinforcement learning loop, including custom neural net definition.
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


4. Miscelaneous notes:  
    * MCTS
        - Use temperature annealing for exploration early in training, low for precise eval later in the optimization 
        - Dirichlet noise: add to the root node during self-play to encourage exploration 
        - Very important: resolve policy search ties at random! 

    - Data augmentation: Augment data using rotation/reflections.



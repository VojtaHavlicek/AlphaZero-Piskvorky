# --- Parameters ---
BOARD_SIZE = 5
WIN_LENGTH = 4

NUM_EPISODES = 10
NUM_WORKERS = 6 # Adjust based on your CPU cores.

DEVICE = "cpu"

# ---- SELF-PLAY PARAMETERS ----
NUM_SELF_PLAY_GAMES = 10 # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
NUM_SELF_PLAY_SIMULATIONS = 100  # Number of MCTS simulations per move.
SELF_PLAY_EXPLORATION_CONSTANT = 5.0  # Exploration constant for MCTS
BUFFER_CAPACITY = 10_000

# ---- TRAINING PARAMETERS ----
BATCH_SIZE = 256
BATCHES_PER_EPISODE = 3 # How many batches to train on each episode (roughly same as the buffer capacity)
NUM_EPOCHS = 3 # How many epochs to train on each batch of data? 
LEARNING_RATE = 1e-3 # Initial learning rate for the optimizer
MODEL_DIR = "models"

# ---- EVAL PARAMETERS ----
EVALUATION_GAMES = 50
NUM_EVAL_SIMULATIONS = 150  # Number of MCTS simulations for evaluation
EVAL_EXPLORATION_CONSTANT = 2.0  # Exploration constant for MCTS during evaluation
EVAL_TEMPERATURE = 0.0  # Temperature for evaluation (lower means more greedy)
# --- Parameters ---
BOARD_SIZE = 5
WIN_LENGTH = 4

NUM_EPISODES = 10
NUM_WORKERS = 6 # Adjust based on your CPU cores.

DEVICE = "cpu"

# ---- GAMEPLAY CONSTANTS ----
X = "X"
O = "O"
DRAW = "D"

# ---- SELF-PLAY PARAMETERS ----
NUM_SELF_PLAY_GAMES = 100 # 100-500 for TicTacToe, 1_000-10_000 for Gomoku
NUM_SELF_PLAY_SIMULATIONS = 200  # Number of MCTS simulations per move.
SELF_PLAY_EXPLORATION_CONSTANT = 2.0  # Exploration constant for MCTS
BUFFER_CAPACITY = 40_000
TEMPERATURE_SCHEDULE_HALFTIME = 100
TEMPERATURE_BASELINE = 0.01  # Base temperature for self-play

# ---- TRAINING PARAMETERS ----
BATCH_SIZE = 1024
BATCHES_PER_EPISODE = 10 # How many batches to train on each episode
NUM_EPOCHS = 3 # How many epochs to train on each batch of data? 
LEARNING_RATE = 5*1e-5# Initial learning rate for the optimizer
MODEL_DIR = "models"

# ---- EVAL PARAMETERS ----
EVALUATION_GAMES = 51
NUM_EVAL_SIMULATIONS = 200  # Number of MCTS simulations for evaluation
EVAL_EXPLORATION_CONSTANT = 2.0  # Exploration constant for MCTS during evaluation
EVAL_TEMPERATURE = 0.3 # Temperature for evaluation (lower means more greedy)
EVAL_TEMPERATURE_SCHEDULE_HALFTIME = 4  # Temperature decay for evaluation
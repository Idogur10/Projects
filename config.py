"""Configuration parameters for trajectory prediction model."""

# Model Architecture
HIDDEN_DIM = 128
NUM_LAYERS = 2
INPUT_DIM = 3  # Cartesian position of the decoder input
OUTPUT_DIM = 6  # velocity and acceleration of the decoder's output

# Training
BATCH_SIZE = 512
N_EPOCHS = 200
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Loss weights
W_POS_ADE = 500.0
W_VEL_MSE = 1000.0
W_ACC_MSE = 20

# Early stopping
PATIENCE = 10

# Data
DIRECTORY = r'C:\Users\idogu\OneDrive\Documents\Master\data'
TRAJECTORY_LENGTH = 250  # sample size
WINDOW_SIZE = 200  # input size K
STEP_SIZE = 50
HORIZON = 50
DELTA_T = 0.01
OUTPUT_SIZE = 9

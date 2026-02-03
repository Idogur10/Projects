"""Configuration parameters for trajectory prediction model."""

# Model Architecture
HIDDEN_DIM = 96
NUM_LAYERS = 1
INPUT_DIM = 3  # Cartesian position of the decoder input
OUTPUT_DIM = 6  # velocity and acceleration of the decoder's output

# Training
BATCH_SIZE = 512
N_EPOCHS = 500
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
TRAJECTORY_LENGTH = 250  # original sample size

# Downsampling: 100Hz -> 10Hz
DOWNSAMPLE_FACTOR = 10
ORIGINAL_WINDOW_SIZE = 200  # original input size at 100Hz
ORIGINAL_HORIZON = 50       # original horizon at 100Hz

# After downsampling (10Hz)
WINDOW_SIZE = ORIGINAL_WINDOW_SIZE // DOWNSAMPLE_FACTOR  # 20 timesteps
HORIZON = ORIGINAL_HORIZON // DOWNSAMPLE_FACTOR          # 5 timesteps
DELTA_T = 0.01 * DOWNSAMPLE_FACTOR                       # 0.1 seconds

STEP_SIZE = 50
OUTPUT_SIZE = 9

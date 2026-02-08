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

# Bi-level optimization (B-spline trajectory fitting)
BSPLINE_BATCH_SIZE = 5      # Batch size for bi-level optimization
BSPLINE_INNER_STEPS = 3     # Number of inner optimization steps
BSPLINE_INNER_LR = 1e-3     # Inner loop learning rate
BSPLINE_OUTER_LR = 5e-4     # Outer loop learning rate (for lambda optimization) - increased for faster convergence
BSPLINE_N_EPOCHS = 2000     # Number of outer optimization epochs
BSPLINE_K = 5               # Number of B-spline control points
BSPLINE_DEGREE = 3          # B-spline degree (3 = cubic)

# Initial hyperparameter values for loss weights
LAMBDA_1_INIT = 500.0       # Position loss weight
LAMBDA_2_INIT = 1000.0      # Power law loss weight
LAMBDA_3_INIT = 1e-7        # Jerk loss weight

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5   # Reduce LR by this factor when plateau detected
LR_SCHEDULER_PATIENCE = 50  # Wait this many epochs before reducing LR
LR_MIN = 1e-6               # Minimum learning rate

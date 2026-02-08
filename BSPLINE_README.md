# B-spline Bi-level Optimization

This directory contains a bi-level optimization pipeline for fitting B-spline trajectories that obey the motor invariant power law (velocity-curvature relationship).

## Project Structure

The previous standalone `Pipeline.py` has been reorganized into modular components:

### Core Modules

- **`models/bspline_optimization.py`**: Contains `ModelCopt` class for inner optimization loop
  - Optimizes control points C for each trajectory to minimize position error, power law error, and jerk
  - Implements differentiable gradient descent to maintain gradient flow for meta-learning

- **`utils/bspline.py`**: B-spline basis matrix initialization
  - `initialize_bspline_matrices()`: Creates B, B_dot, B_ddot, B_dddot matrices for position, velocity, acceleration, and jerk

- **`utils/power_law.py`**: Power law parameter extraction
  - `get_parameters_per_sample()`: Fits V = α * κ^β for each trajectory using log-linear regression

- **`data/preprocessing.py`**: Data preprocessing utilities
  - `downsample_data()`: Downsamples trajectory data (e.g., 100Hz → 10Hz)
  - `preprocessing()`: Centers positions and scales velocities

### Configuration

All bi-level optimization hyperparameters are in **`config.py`**:

```python
# Bi-level optimization settings
BSPLINE_BATCH_SIZE = 5       # Number of trajectories per batch
BSPLINE_INNER_STEPS = 3      # Inner optimization steps per trajectory
BSPLINE_INNER_LR = 1e-3      # Inner loop learning rate
BSPLINE_OUTER_LR = 1e-4      # Outer loop learning rate (lambda optimization)
BSPLINE_N_EPOCHS = 50        # Number of outer optimization epochs
BSPLINE_K = 5                # Number of B-spline control points
BSPLINE_DEGREE = 3           # B-spline degree (3 = cubic)

# Initial loss weights
LAMBDA_1_INIT = 500.0        # Position loss weight
LAMBDA_2_INIT = 1000.0       # Power law loss weight
LAMBDA_3_INIT = 1e-7         # Jerk loss weight
```

### Main Training Script

**`train_bspline.py`**: Main entry point for bi-level optimization

Run with:
```bash
python train_bspline.py
```

## Algorithm Overview

### Bi-level Optimization

**Inner Loop (per trajectory):**
For each trajectory i, optimize control points C_i to minimize:
```
L_inner = λ₁ * ||P(s) - R_U||² + λ₂ * ||V - α*κ^β||² + λ₃ * ||J||²
```

Where:
- P(s) = B @ C_i (position from B-spline)
- V(s) = B_dot @ C_i (velocity)
- κ(s) = curvature = ||v × a|| / ||v||³
- J(s) = B_dddot @ C_i (jerk)
- α, β = power law parameters per trajectory

**Outer Loop:**
Optimize hyperparameters λ₁, λ₂, λ₃ to minimize trajectory reconstruction error:
```
L_outer = mean(||P_true - P_hat||)
```

Uses `create_graph=True` in inner loop gradient computation to maintain gradient flow from outer loss through inner optimization back to λ.

## Key Technical Details

### Gradient Flow
- Inner optimization uses **differentiable gradient descent** (no `torch.no_grad()`)
- `create_graph=True` keeps computational graph for meta-learning
- Enables λ gradients to flow through inner optimization steps

### Numerical Stability
- Power law computation in log-space: κ^β = exp(β * log(κ))
- Extensive clamping to prevent overflow/underflow
- Hyperparameters in log-space to ensure positivity

### Working Configuration
Current stable configuration (from successful runs):
- Batch size: 5 trajectories
- Inner steps: 3
- Inner LR: 1e-3
- λ₁ and λ₂ learn successfully
- λ₃ gradient typically zero (expected due to downsampling and B-spline smoothness)

## Backup

The original standalone script is preserved as `Pipeline_old.py` for reference.

## Dependencies

All required packages are listed in `requirements.txt`. Key dependencies:
- PyTorch (with CUDA support)
- NumPy
- SciPy (for B-spline basis functions)
- scikit-learn (for preprocessing)

## Future Work

Potential improvements:
- Increase control points K to enable more complex trajectories
- Experiment with different B-spline degrees
- Add validation set evaluation
- Visualize learned trajectories vs. ground truth

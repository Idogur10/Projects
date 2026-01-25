# Trajectory Prediction using Physics-Informed Seq2Seq GRU

A deep learning approach for trajectory prediction using a sequence-to-sequence GRU architecture with physics-based motion integration.

## Overview

This project implements a trajectory prediction model that combines:
- **GRU Encoder-Decoder Architecture**: Captures temporal dependencies in motion sequences
- **Physics-Based Integration**: Uses Velocity Verlet integration for physically plausible predictions
- **Teacher Forcing**: Improves training stability with scheduled sampling

The model predicts future 3D positions, velocities, and accelerations given a history of observed trajectory points.

## Project Structure

```
├── config.py                 # Hyperparameters and settings
├── train.py                  # Main training script
├── data/
│   ├── dataset.py            # PyTorch Dataset class
│   └── preprocessing.py      # Data normalization functions
├── models/
│   └── seq2seq.py            # Encoder, Decoder, and Seq2Seq model
├── utils/
│   ├── losses.py             # ADE loss function
│   ├── evaluation.py         # Evaluation metrics (MAE, RMSE, L2)
│   └── visualization.py      # 3D trajectory plotting
└── requirements.txt          # Python dependencies
```

## Model Architecture

```
Input Sequence (200 steps)
        │
        ▼
┌───────────────┐
│  GRU Encoder  │  ──►  Hidden State
└───────────────┘
        │
        ▼
┌───────────────┐
│  GRU Decoder  │  +  Velocity Verlet Physics
└───────────────┘
        │
        ▼
Predicted Trajectory (50 steps)
   [position, velocity, acceleration]
```

### Key Components

| Component | Description |
|-----------|-------------|
| `Encoder_GRU` | Encodes input trajectory into hidden representation |
| `Decoder_GRU` | Autoregressively generates future predictions |
| `Seq2SeqGRU` | Combines encoder/decoder with physics integration |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Idogur10/Projects.git
cd Projects
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Configuration

Edit `config.py` to modify hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | 128 | GRU hidden layer size |
| `NUM_LAYERS` | 2 | Number of stacked GRU layers |
| `BATCH_SIZE` | 512 | Training batch size |
| `N_EPOCHS` | 200 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Adam optimizer learning rate |
| `WINDOW_SIZE` | 200 | Input sequence length |
| `HORIZON` | 50 | Prediction horizon |
| `DELTA_T` | 0.01 | Time step (seconds) |

## Usage

### Training

```bash
python train.py
```

The training script will:
1. Load and preprocess trajectory data
2. Train the model with early stopping
3. Evaluate on validation/test sets
4. Generate visualization plots
5. Save the best model to `best_model.pth`

### Data Format

Input data should be `.npy` files with shape `(N, sequence_length, 13)`:
- Columns 0-2: Position (X, Y, Z)
- Columns 3-5: Velocity (Vx, Vy, Vz)
- Columns 6-8: Acceleration (Ax, Ay, Az)
- Columns 9-12: Additional features

## Evaluation Metrics

The model is evaluated using:
- **ADE (Average Displacement Error)**: Mean L2 distance between predicted and true positions
- **MAE**: Mean Absolute Error per axis (X, Y, Z)
- **RMSE**: Root Mean Square Error per axis
- **L2 Distance**: Euclidean distance at specific timesteps (10, 20, 30, 40, 50)

Results are reported in millimeters (mm).

## Loss Function

Combined weighted loss:
```
Loss = W_POS × ADE + W_VEL × MSE_velocity + W_ACC × MSE_acceleration
```

Default weights: `W_POS=500`, `W_VEL=1000`, `W_ACC=20`

## Results

Training produces:
- Loss curves (train/validation)
- 3D trajectory visualizations
- Per-step error plots (MAE, RMSE)
- Evaluation metrics at prediction horizons

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Author

Ido Gur

## License

This project is part of a Master's thesis.

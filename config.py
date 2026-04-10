"""
config.py - Central configuration for the Seizure Detection project.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "BEED_Data.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
NUM_FEATURES = 16         # X1 .. X16  (EEG channels)
NUM_CLASSES = 4           # 0, 1, 2, 3
CLASS_NAMES = [
    "Normal",             # 0
    "Interictal",         # 1
    "Pre-ictal",          # 2
    "Seizure",            # 3
]
TEST_SIZE = 0.2           # 80-20 train/test split
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
SCALER_TYPE = "standard"  # "standard" | "minmax"

# ──────────────────────────────────────────────
# Model / Training
# ──────────────────────────────────────────────
HIDDEN_SIZES = [128, 64, 32]   # MLP hidden layers
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 8        # stop if val-loss doesn't improve

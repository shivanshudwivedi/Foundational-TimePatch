"""
Central configuration file for the Asteroid Transformer project.
"""

import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Project Paths ---
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs") # <-- ADD THIS

# --- Data & Pre-processing ---
FEATURES = ["x", "y", "z", "vx", "vy", "vz"]
NUM_FEATURES = len(FEATURES)
N_ASTEROIDS = 5           # Number of asteroids we are training on 
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Model Hyperparameters ---
PATCH_LEN = 4
INPUT_STEPS = 24
OUTPUT_STEPS = 6
N_PATCHES = INPUT_STEPS // PATCH_LEN

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 6
D_FF = D_MODEL * 4
DROPOUT = 0.1
ACTIVATION = "gelu"

EMBED_DIM_ASTEROID = 16

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 100
PATIENCE = 10

# --- Evaluation & Checkpointing ---
BASELINE_REPORT = os.path.join(ARTIFACT_DIR, "evaluation_report.json")
TRANSFORMER_REPORT = os.path.join(ARTIFACT_DIR, "transformer_report.json")
TRANSFORMER_PLOT = os.path.join(ARTIFACT_DIR, "transformer_forecast.png")
# We replace MODEL_SAVE_PATH with this:
BEST_CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "best_model.ckpt")


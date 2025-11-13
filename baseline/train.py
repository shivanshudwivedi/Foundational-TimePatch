"""
Trains the two-stage baseline FNN model, as described in the
LEO orbit prediction paper (arXiv:2407.11026v1).

This involves two steps:
1.  Train a "CoarseFNN" on only the asteroid's state vectors (x,y,z,vx,vy,vz).
2.  Train a "RefinementFNN" whose inputs are:
    a) The predictions from the CoarseFNN.
    b) The exogenous features (past and future Sun/Jupiter positions).

Input:
    datasets/train.npz, datasets/val.npz

Output:
    artifacts/coarse_fnn.pt (Model from Stage 1)
    artifacts/refinement_fnn.pt (Final model from Stage 2)
    artifacts/coarse_training_curves.png
    artifacts/refinement_training_curves.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
DATASET_DIR  = "datasets"
ARTIFACT_DIR = "artifacts"

BATCH_SIZE   = 128
HIDDEN_DIM   = 256
LR           = 1e-3
EPOCHS       = 100  # Will be used for both training stages
PATIENCE     = 10   # early stopping
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# From our pre-process.py, we know the feature split
NUM_STATE_FEATURES = 6  # (x, y, z, vx, vy, vz)
NUM_EXOG_FEATURES  = 6  # (x_sun, y_sun, z_sun, x_jup, y_jup, z_jup)

print(f"Using device: {DEVICE}")

# ----------------------------
# Model definition
# ----------------------------
class FNNBaseline(nn.Module):
    """Reusable FNN model class."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Reusable Training Loop
# ----------------------------
def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                save_path: str,
                plot_path: str,
                epochs: int, 
                patience: int, 
                lr: float):
    """
    Generic training loop for an FNN model.
    Saves the best model based on validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    
    print(f"\n--- Starting training for: {os.path.basename(save_path)} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[early stop] No improvement for {patience} epochs.")
                break

    print(f"[done] Best val MSE: {best_val_loss:.6f}")
    print(f"[saved] {save_path}")

    # Plot training curve
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{os.path.basename(save_path)} Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[saved] {plot_path}")
    return best_val_loss

# ----------------------------
# Load and Split Datasets
# ----------------------------
def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["Y"]

print("[info] Loading datasets...")
X_train_full, Y_train_full = load_npz(os.path.join(DATASET_DIR, "train.npz"))
X_val_full,   Y_val_full   = load_npz(os.path.join(DATASET_DIR, "val.npz"))

input_steps  = int(X_train_full.shape[1])
output_steps = int(Y_train_full.shape[1])
num_total_features = int(X_train_full.shape[2])

assert num_total_features == NUM_STATE_FEATURES + NUM_EXOG_FEATURES

print(f"[info] X_train: {X_train_full.shape}, Y_train: {Y_train_full.shape}")

# --- Split data into state (asteroid) and exog (Sun/Jupiter) ---
# State data (what we want to predict)
X_train_state = X_train_full[:, :, :NUM_STATE_FEATURES]
Y_train_state = Y_train_full[:, :, :NUM_STATE_FEATURES]
X_val_state   = X_val_full[:, :, :NUM_STATE_FEATURES]
Y_val_state   = Y_val_full[:, :, :NUM_STATE_FEATURES]

# Exogenous data (our "physics" features)
X_train_exog = X_train_full[:, :, NUM_STATE_FEATURES:]
Y_train_exog = Y_train_full[:, :, NUM_STATE_FEATURES:]
X_val_exog   = X_val_full[:, :, NUM_STATE_FEATURES:]
Y_val_exog   = Y_val_full[:, :, NUM_STATE_FEATURES:]

# ----------------------------------------------------
# STAGE 1: Train CoarseFNN
# ----------------------------------------------------
print("\n=== STAGE 1: Training CoarseFNN ===")
# This model only sees the asteroid's past state
# and only predicts the asteroid's future state.

coarse_input_dim  = input_steps * NUM_STATE_FEATURES
coarse_output_dim = output_steps * NUM_STATE_FEATURES

coarse_model = FNNBaseline(input_dim=coarse_input_dim,
                           hidden_dim=HIDDEN_DIM,
                           output_dim=coarse_output_dim).to(DEVICE)

# Flatten data for FNN
X_train_state_f = X_train_state.reshape(len(X_train_state), -1)
Y_train_state_f = Y_train_state.reshape(len(Y_train_state), -1)
X_val_state_f   = X_val_state.reshape(len(X_val_state), -1)
Y_val_state_f   = Y_val_state.reshape(len(Y_val_state), -1)

# Create DataLoaders
train_loader_coarse = DataLoader(
    TensorDataset(torch.tensor(X_train_state_f, dtype=torch.float32),
                  torch.tensor(Y_train_state_f, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True)

val_loader_coarse = DataLoader(
    TensorDataset(torch.tensor(X_val_state_f, dtype=torch.float32),
                  torch.tensor(Y_val_state_f, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False)

# Train the Coarse Model
train_model(coarse_model, 
            train_loader_coarse, 
            val_loader_coarse,
            save_path=os.path.join(ARTIFACT_DIR, "coarse_fnn.pt"),
            plot_path=os.path.join(ARTIFACT_DIR, "coarse_training_curves.png"),
            epochs=EPOCHS, 
            patience=PATIENCE, 
            lr=LR)

# ----------------------------------------------------
# STAGE 2: Train RefinementFNN
# ----------------------------------------------------
print("\n=== STAGE 2: Training RefinementFNN ===")
# This model sees the CoarseFNN's guess + all exogenous data

# 1. Load the best CoarseFNN weights
coarse_model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "coarse_fnn.pt")))
coarse_model.eval()

# 2. Get coarse predictions for the *entire* train and val sets
print("[info] Generating coarse predictions for refinement model...")
with torch.no_grad():
    X_train_state_t = torch.tensor(X_train_state_f, dtype=torch.float32).to(DEVICE)
    X_val_state_t   = torch.tensor(X_val_state_f, dtype=torch.float32).to(DEVICE)
    
    coarse_preds_train_t = coarse_model(X_train_state_t) # (N_train, output_steps * 6)
    coarse_preds_val_t   = coarse_model(X_val_state_t)   # (N_val,   output_steps * 6)

# 3. Flatten exogenous data
X_train_exog_f_t = torch.tensor(X_train_exog.reshape(len(X_train_exog), -1), dtype=torch.float32).to(DEVICE)
Y_train_exog_f_t = torch.tensor(Y_train_exog.reshape(len(Y_train_exog), -1), dtype=torch.float32).to(DEVICE)
X_val_exog_f_t   = torch.tensor(X_val_exog.reshape(len(X_val_exog), -1), dtype=torch.float32).to(DEVICE)
Y_val_exog_f_t   = torch.tensor(Y_val_exog.reshape(len(Y_val_exog), -1), dtype=torch.float32).to(DEVICE)

# 4. Create the NEW combined input for the RefinementFNN
# Input = [coarse_guess | past_exog | future_exog]
X_refine_train_t = torch.cat([coarse_preds_train_t, X_train_exog_f_t, Y_train_exog_f_t], dim=1)
X_refine_val_t   = torch.cat([coarse_preds_val_t,   X_val_exog_f_t,   Y_val_exog_f_t],   dim=1)

# The TARGET is the true asteroid state
Y_refine_train_t = torch.tensor(Y_train_state_f, dtype=torch.float32).to(DEVICE)
Y_refine_val_t   = torch.tensor(Y_val_state_f, dtype=torch.float32).to(DEVICE)

print(f"[info] Refinement model input shape: {X_refine_train_t.shape}")
print(f"[info] Refinement model output shape: {Y_refine_train_t.shape}")

# 5. Define the RefinementFNN
refine_input_dim  = X_refine_train_t.shape[1]
refine_output_dim = Y_refine_train_t.shape[1]

refinement_model = FNNBaseline(input_dim=refine_input_dim,
                               hidden_dim=HIDDEN_DIM,
                               output_dim=refine_output_dim).to(DEVICE)

# 6. Create new DataLoaders
train_loader_refine = DataLoader(
    TensorDataset(X_refine_train_t, Y_refine_train_t),
    batch_size=BATCH_SIZE, shuffle=True)

val_loader_refine = DataLoader(
    TensorDataset(X_refine_val_t, Y_refine_val_t),
    batch_size=BATCH_SIZE, shuffle=False)

# 7. Train the final Refinement Model
train_model(refinement_model,
            train_loader_refine,
            val_loader_refine,
            save_path=os.path.join(ARTIFACT_DIR, "refinement_fnn.pt"),
            plot_path=os.path.join(ARTIFACT_DIR, "refinement_training_curves.png"),
            epochs=EPOCHS,
            patience=PATIENCE,
            lr=LR)

print("\n--- Both models trained successfully! ---")
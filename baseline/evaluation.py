"""
Evaluates the trained two-stage FNN baseline on the test split.

Inputs:
    datasets/test.npz
    artifacts/coarse_fnn.pt
    artifacts/refinement_fnn.pt
    artifacts/scaler.joblib

Outputs:
    artifacts/evaluation_report.json
    artifacts/sample_forecast.png
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------
# Config
# ----------------------------
DATASET_DIR  = "datasets"
ARTIFACT_DIR = "artifacts"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_INDEX = 0   # index of a test example to visualize

# From our pre-process.py, we know the feature split
NUM_STATE_FEATURES = 6  # (x, y, z, vx, vy, vz)
NUM_EXOG_FEATURES  = 6  # (x_sun, y_sun, z_sun, x_jup, y_jup, z_jup)

print(f"Using device: {DEVICE}")

# ----------------------------
# Load dataset & scaler
# ----------------------------
def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["Y"], data["input_steps"].item(), data["output_steps"].item(), list(data["features"])

X_test, Y_test, input_steps, output_steps, features = load_npz(os.path.join(DATASET_DIR, "test.npz"))
scaler = load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))

num_total_features = len(features)
assert num_total_features == NUM_STATE_FEATURES + NUM_EXOG_FEATURES

print(f"[info] Loaded test set: {X_test.shape} | features: {features}")

# ----------------------------
# Define model (same as training)
# ----------------------------
class FNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Load BOTH trained models
# ----------------------------

# 1. Define and load CoarseFNN
coarse_input_dim  = input_steps * NUM_STATE_FEATURES
coarse_output_dim = output_steps * NUM_STATE_FEATURES

coarse_model = FNNBaseline(input_dim=coarse_input_dim,
                           hidden_dim=256,
                           output_dim=coarse_output_dim).to(DEVICE)
coarse_model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "coarse_fnn.pt"), map_location=DEVICE))
coarse_model.eval()
print("[ok] Loaded CoarseFNN model weights.")

# 2. Define and load RefinementFNN
refine_input_dim = coarse_output_dim + (input_steps * NUM_EXOG_FEATURES) + (output_steps * NUM_EXOG_FEATURES)
refine_output_dim = coarse_output_dim

refinement_model = FNNBaseline(input_dim=refine_input_dim,
                               hidden_dim=256,
                               output_dim=refine_output_dim).to(DEVICE)
refinement_model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "refinement_fnn.pt"), map_location=DEVICE))
refinement_model.eval()
print("[ok] Loaded RefinementFNN model weights.")

# ----------------------------
# Prepare data for 2-stage prediction
# ----------------------------

# Split test set into state (asteroid) and exog (Sun/Jupiter)
X_test_state = X_test[:, :, :NUM_STATE_FEATURES]
X_test_exog  = X_test[:, :, NUM_STATE_FEATURES:]
Y_test_exog  = Y_test[:, :, NUM_STATE_FEATURES:]

# Ground truth is the future state
Y_test_state = Y_test[:, :, :NUM_STATE_FEATURES]

# Flatten all components for FNN inputs
X_test_state_f = X_test_state.reshape(len(X_test_state), -1)
X_test_exog_f  = X_test_exog.reshape(len(X_test_exog), -1)
Y_test_exog_f  = Y_test_exog.reshape(len(Y_test_exog), -1)

# Convert to tensors
X_test_state_t = torch.tensor(X_test_state_f, dtype=torch.float32).to(DEVICE)
X_test_exog_f_t  = torch.tensor(X_test_exog_f, dtype=torch.float32).to(DEVICE)
Y_test_exog_f_t  = torch.tensor(Y_test_exog_f, dtype=torch.float32).to(DEVICE)

# ----------------------------
# Predict (2-Stage)
# ----------------------------
with torch.no_grad():
    # Stage 1: Get coarse predictions
    coarse_preds_t = coarse_model(X_test_state_t)
    
    # Stage 2: Create input for refinement model
    # Input = [coarse_guess | past_exog | future_exog]
    X_refine_input_t = torch.cat([coarse_preds_t, X_test_exog_f_t, Y_test_exog_f_t], dim=1)
    
    # Get final predictions (predicted future state)
    final_preds_t = refinement_model(X_refine_input_t)

# Get final predictions and ground truth as numpy arrays
# These are the *normalized* state predictions, shape (N, output_steps * 6)
preds_f = final_preds_t.cpu().numpy()
true_f  = Y_test_state.reshape(len(Y_test_state), -1)

# ----------------------------
# Inverse transform to physical units
# ----------------------------
# This is the crucial part. The scaler was fit on ALL features.
# We must "reconstruct" the full 12-feature array before inverse_transform.

# 1. Reshape predictions to (N, output_steps, num_state_features)
preds_reshaped = preds_f.reshape(len(Y_test), output_steps, NUM_STATE_FEATURES)

# 2. Re-combine with the (known) future exogenous features
#    Y_test_exog shape is (N, output_steps, num_exog_features)
Y_pred_full_norm = np.concatenate([preds_reshaped, Y_test_exog], axis=2)
Y_true_full_norm = Y_test # This is already the full (N, output_steps, 12) array

# 3. Flatten both to (N * steps, F) for scaler.inverse_transform
Y_pred_flat_norm = Y_pred_full_norm.reshape(-1, num_total_features)
Y_true_flat_norm = Y_true_full_norm.reshape(-1, num_total_features)

# 4. Inverse transform to get physical units [AU, AU/day]
Y_pred_inv = scaler.inverse_transform(Y_pred_flat_norm)
Y_true_inv = scaler.inverse_transform(Y_true_flat_norm)

# ----------------------------
# Compute metrics on *state features only*
# ----------------------------
# We only care about the error on the 6 features we predicted.
Y_true_inv_state = Y_true_inv[:, :NUM_STATE_FEATURES]
Y_pred_inv_state = Y_pred_inv[:, :NUM_STATE_FEATURES]

rmse = np.sqrt(mean_squared_error(Y_true_inv_state, Y_pred_inv_state))
mae  = mean_absolute_error(Y_true_inv_state, Y_pred_inv_state)
print(f"\n--- Evaluation Metrics (on predicted state features) ---")
print(f"Test RMSE: {rmse:.6f} [Physical Units]")
print(f"Test MAE:  {mae:.6f} [Physical Units]")

# ----------------------------
# Save numeric report
# ----------------------------
report = {
    "model_architecture": "Two-Stage FNN (Coarse + Refinement)",
    "test_rmse_state_only": float(rmse),
    "test_mae_state_only": float(mae),
    "num_test_samples": int(len(X_test)),
    "input_steps": int(input_steps),
    "output_steps": int(output_steps),
    "features": features,
    "num_state_features": NUM_STATE_FEATURES,
    "num_exog_features": NUM_EXOG_FEATURES
}
report_path = os.path.join(ARTIFACT_DIR, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"[saved] {report_path}")

# ----------------------------
# Plot a sample forecast
# ----------------------------
idx = min(SAMPLE_INDEX, len(X_test) - 1)

# Get the (output_steps, 12) block for the sample
true_seq = Y_true_inv[idx*output_steps:(idx+1)*output_steps]
pred_seq = Y_pred_inv[idx*output_steps:(idx+1)*output_steps]

plt.figure(figsize=(8,5))
# Plot x (col 0) vs y (col 1)
plt.plot(true_seq[:,0], true_seq[:,1], 'o-', label="True (XY)")
plt.plot(pred_seq[:,0], pred_seq[:,1], 'x--', label="Predicted (XY)")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title(f"Sample Forecasted Orbit Segment (idx {idx}) - XY Plane")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(ARTIFACT_DIR, "sample_forecast.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[saved] {plot_path}")
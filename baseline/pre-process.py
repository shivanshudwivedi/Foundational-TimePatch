"""
Reads an asteroid catalog, fetches data from JPL Horizons,
and builds normalized, sliding-window (X, y) datasets.

This version (v2.1) fixes a datetime parsing error for
"A.D." strings from JPL Horizons.

It fetches:
1.  Target Asteroid state vectors (x, y, z, vx, vy, vz)
2.  Exogenous Perturber state vectors (Sun, Jupiter)

It saves:
- datasets/train.npz, val.npz, test.npz
- artifacts/scaler.joblib
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import dump
from typing import Tuple, List, Optional
from time import sleep

# --- Import astroquery ---
try:
    from astroquery.jplhorizons import Horizons
except ImportError:
    print("Error: 'astroquery' package not found.")
    print("Please install it: pip install astroquery")
    exit(1)


# ----------------------------
# Config
# ----------------------------
# --- Data Sources ---
CATALOG_PATH  = "./data/raw/dataset.csv"   # Your asteroid catalog
N_ASTEROIDS   = 5                        # Fetch first N asteroids from catalog

# --- JPL Horizons Config ---
START_DATE    = "2024-10-02"
STOP_DATE     = "2024-12-01"
STEP_SIZE     = "1h"
CENTER        = "@0"     # Solar System Barycenter
REFPLANE      = "ecliptic"

# --- Pre-processing Config ---
ARTIFACT_DIR  = "artifacts"
DATASET_DIR   = "datasets"

INPUT_STEPS   = 24      # e.g., 24 hours of history
OUTPUT_STEPS  = 6       # e.g., predict next 6 hours

# Define all features, including exogenous
FEATURES      = [
    "x", "y", "z", "vx", "vy", "vz",  # Asteroid
    "x_sun", "y_sun", "z_sun",        # Exogenous: Sun
    "x_jup", "y_jup", "z_jup"         # Exogenous: Jupiter
]

# --- Splitting Config ---
VAL_RATIO     = 0.1     # last 10% of timesteps (per asteroid) for val
TEST_RATIO    = 0.1     # last 10% (after val) for test
MIN_LENGTH    = INPUT_STEPS + OUTPUT_STEPS + 20  # skip too-short series

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR,  exist_ok=True)

# ----------------------------
# Data Fetching
# ----------------------------
def fetch_combined_vectors(identifier: str, start: str, stop: str, step: str, 
                           center: str, refplane: str) -> Optional[pd.DataFrame]:
    """
    Fetches state vectors for a target asteroid AND main perturbers
    (Sun, Jupiter) and returns a combined DataFrame.
    """
    try:
        # 1. Fetch Target Asteroid
        obj_ast = Horizons(
            id=str(identifier),
            id_type="smallbody",
            location=center,
            epochs={'start': start, 'stop': stop, 'step': step}
        )
        vec_ast = obj_ast.vectors(refplane=refplane).to_pandas()
        
        # --- FIX IS HERE ---
        # Use .copy() to avoid SettingWithCopyWarning
        df_ast = vec_ast[['datetime_str','x','y','z','vx','vy','vz']].copy()
        df_ast.rename(columns={'datetime_str':'datetime'}, inplace=True)
        # Remove "A.D. " prefix before parsing
        df_ast['datetime'] = pd.to_datetime(df_ast['datetime'].str.replace("A.D. ", ""))
        # --- END FIX ---

        # 2. Fetch Sun (@10)
        obj_sun = Horizons(
            id='10', id_type='majorbody', location=center,
            epochs={'start': start, 'stop': stop, 'step': step}
        )
        vec_sun = obj_sun.vectors(refplane=refplane).to_pandas()
        df_sun = vec_sun[['x','y','z']].rename(
            columns={'x':'x_sun', 'y':'y_sun', 'z':'z_sun'}
        )

        # 3. Fetch Jupiter (@5)
        obj_jup = Horizons(
            id='5', id_type='majorbody', location=center,
            epochs={'start': start, 'stop': stop, 'step': step}
        )
        vec_jup = obj_jup.vectors(refplane=refplane).to_pandas()
        df_jup = vec_jup[['x','y','z']].rename(
            columns={'x':'x_jup', 'y':'y_jup', 'z':'z_jup'}
        )
        
        # 4. Combine
        # Use index alignment (should be identical from Horizons)
        df_combined = pd.concat([df_ast, df_sun, df_jup], axis=1)
        df_combined = df_combined.sort_values('datetime').reset_index(drop=True)
        
        # Check for missing data
        if df_combined.isnull().values.any():
            print(f"[warn] NaNs found in fetched data for {identifier}. Dropping NaNs.")
            df_combined = df_combined.dropna().reset_index(drop=True)
            
        return df_combined

    except Exception as e:
        print(f"[error] Failed to fetch data for asteroid {identifier}: {e}")
        return None

# ----------------------------
# Pre-processing Helpers
# ----------------------------
def create_sliding_windows(arr: np.ndarray, input_steps: int, output_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates (X, Y) windows from a time-series array."""
    X, Y = [], []
    T = len(arr)
    for i in range(T - input_steps - output_steps + 1):
        X.append(arr[i:i+input_steps])
        Y.append(arr[i+input_steps:i+input_steps+output_steps])
    return np.asarray(X), np.asarray(Y)

def time_split_indices(T: int, val_ratio: float, test_ratio: float) -> Tuple[int,int]:
    """Returns (val_start, test_start) indices for a time-ordered split."""
    val_start  = int(np.floor(T * (1.0 - (val_ratio + test_ratio))))
    test_start = int(np.floor(T * (1.0 - test_ratio)))
    return val_start, test_start

# ----------------------------
# Main build
# ----------------------------
def main():
    # 1) Load asteroid catalog
    try:
        catalog = pd.read_csv(CATALOG_PATH)
    except FileNotFoundError:
        print(f"[error] Catalog file not found at {CATALOG_PATH}")
        print("Please run fetch-data.ipynb or provide the correct path.")
        return
    except pd.errors.DtypeWarning as e:
        print(f"[warn] DtypeWarning reading catalog: {e}")
        print("       Attempting to load with low_memory=False")
        catalog = pd.read_csv(CATALOG_PATH, low_memory=False)


    # Select asteroids
    asteroids = catalog.head(N_ASTEROIDS)
    print(f"--- Processing {len(asteroids)} asteroids from {CATALOG_PATH} ---")

    # 2) Loop 1: Fetch all data and fit scaler
    all_feature_rows: List[np.ndarray] = []
    fetched_series_data: List[pd.DataFrame] = []

    for _, row in asteroids.iterrows():
        ident = row['pdes'] if not pd.isna(row['pdes']) else row['full_name']
        name = row['name'] if pd.notna(row['name']) else row['full_name']
        
        print(f"\n[fetch] Fetching {ident} ({name})...")
        df = fetch_combined_vectors(ident, START_DATE, STOP_DATE, STEP_SIZE, CENTER, REFPLANE)
        sleep(1) # Be polite to JPL Horizons

        if df is None:
            continue

        # Validate
        missing_cols = [f for f in FEATURES if f not in df.columns]
        if missing_cols:
            print(f"[skip] {name} missing required cols: {missing_cols}")
            continue
        if len(df) < MIN_LENGTH:
            print(f"[skip] {name} too short: {len(df)} rows (min {MIN_LENGTH})")
            continue

        feat_values = df[FEATURES].values.astype(np.float64)
        all_feature_rows.append(feat_values)
        fetched_series_data.append(df) # Save the whole df for loop 2
        print(f"[ok] {name} loaded ({len(df)} timesteps)")

    if not fetched_series_data:
        raise RuntimeError("No valid asteroid series available after fetching.")

    # Fit scaler on ALL data from ALL asteroids
    stacked_features = np.vstack(all_feature_rows)  # (total_T_all_asteroids, num_features)
    scaler = StandardScaler().fit(stacked_features)
    dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    print(f"\n[ok] Fitted scaler on {stacked_features.shape[0]} total rows.")
    print(f"      Saved: artifacts/scaler.joblib")

    # 3) Loop 2: Build windows per asteroid, split by time, and concat pools
    X_tr_list, Y_tr_list = [], []
    X_va_list, Y_va_list = [], []
    X_te_list, Y_te_list = [], []

    print("\n--- Building Train/Val/Test windows ---")
    for df in fetched_series_data:
        feat_values = df[FEATURES].values.astype(np.float64)
        feat_norm = scaler.transform(feat_values)

        # Split normalized array by time
        val_start, test_start = time_split_indices(len(feat_norm), VAL_RATIO, TEST_RATIO)
        tr_norm = feat_norm[:val_start]
        va_norm = feat_norm[val_start:test_start]
        te_norm = feat_norm[test_start:]

        # Create windows for each split
        Xtr, Ytr = create_sliding_windows(tr_norm, INPUT_STEPS, OUTPUT_STEPS)
        Xva, Yva = create_sliding_windows(va_norm, INPUT_STEPS, OUTPUT_STEPS)
        Xte, Yte = create_sliding_windows(te_norm, INPUT_STEPS, OUTPUT_STEPS)

        if len(Xtr): X_tr_list.append(Xtr); Y_tr_list.append(Ytr)
        if len(Xva): X_va_list.append(Xva); Y_va_list.append(Yva)
        if len(Xte): X_te_list.append(Xte); Y_te_list.append(Yte)

        # Get a display name
        name = "Unknown Asteroid"
        if 'full_name' in df.columns:
            name = df['full_name'].iloc[0]
        elif 'name' in df.columns:
            name = df['name'].iloc[0]

        print(f"[windows] {name} → "
              f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")

    # 4) Stack all windows from all asteroids into final datasets
    def _stack_or_empty(L, shape):
        if not L:
            return np.empty(shape, dtype=np.float64)
        return np.vstack(L)

    # Define empty shapes in case a split has 0 samples
    x_empty_shape = (0, INPUT_STEPS, len(FEATURES))
    y_empty_shape = (0, OUTPUT_STEPS, len(FEATURES))

    X_train = _stack_or_empty(X_tr_list, x_empty_shape)
    Y_train = _stack_or_empty(Y_tr_list, y_empty_shape)
    X_val   = _stack_or_empty(X_va_list, x_empty_shape)
    Y_val   = _stack_or_empty(Y_va_list, y_empty_shape)
    X_test  = _stack_or_empty(X_te_list, x_empty_shape)
    Y_test  = _stack_or_empty(Y_te_list, y_empty_shape)

    # 5) Save to .npz
    for name, arr in [("X_train", X_train), ("Y_train", Y_train),
                      ("X_val", X_val), ("Y_val", Y_val),
                      ("X_test", X_test), ("Y_test", Y_test)]:
        if arr.shape[0] == 0:
            print(f"[warn] {name} is empty.")
        else:
            print(f"[ok] {name}: {arr.shape}")

    np.savez_compressed(os.path.join(DATASET_DIR, "train.npz"),
                        X=X_train, Y=Y_train,
                        input_steps=INPUT_STEPS, output_steps=OUTPUT_STEPS,
                        features=FEATURES)
    np.savez_compressed(os.path.join(DATASET_DIR, "val.npz"),
                        X=X_val, Y=Y_val,
                        input_steps=INPUT_STEPS, output_steps=OUTPUT_STEPS,
                        features=FEATURES)
    np.savez_compressed(os.path.join(DATASET_DIR, "test.npz"),
                        X=X_test, Y=Y_test,
                        input_steps=INPUT_STEPS, output_steps=OUTPUT_STEPS,
                        features=FEATURES)
    print(f"\n[saved] train.npz, val.npz, test.npz to {DATASET_DIR}")
    
    # 6) Save manifest
    manifest = {
        "data_catalog": CATALOG_PATH,
        "n_asteroids_processed": len(fetched_series_data),
        "jpl_start_date": START_DATE,
        "jpl_stop_date": STOP_DATE,
        "jpl_step_size": STEP_SIZE,
        "input_steps": INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
        "features": FEATURES,
        "num_features": len(FEATURES),
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": np.sqrt(scaler.var_).tolist(),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(DATASET_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] manifest.json to {DATASET_DIR}")
    print("\n--- Pre-processing complete! ---")

if __name__ == "__main__":
    # Note: Assumes a "data/raw/dataset.csv" file exists.
    # You may need to create this directory and file.
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw", exist_ok=True)
        print("Created 'data/raw' directory.")
        print("Please ensure your 'dataset.csv' catalog is placed there.")
            
    main()
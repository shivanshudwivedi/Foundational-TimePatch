"""
Pre-processes the raw asteroid .csv files (from fetch-data.ipynb)
into windowed .npz files for the Transformer.

This version (v-transformer):
1.  Uses ONLY the 6D state features.
2.  Does NOT scale the data (this will be done by RevIN in the model).
3.  Creates and saves an 'asteroid_ids' array for the embedding layer.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List

# Import from our new config file
import config

# ----------------------------
# Helpers
# ----------------------------

def create_sliding_windows(
    arr: np.ndarray, 
    input_steps: int, 
    output_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates (X, Y) windows from a time-series array."""
    X, Y = [], []
    T = len(arr)
    for i in range(T - input_steps - output_steps + 1):
        X.append(arr[i : i + input_steps])
        Y.append(arr[i + input_steps : i + input_steps + output_steps])
    return np.asarray(X), np.asarray(Y)

def time_split_indices(
    T: int, val_ratio: float, test_ratio: float
) -> Tuple[int, int]:
    """Returns (val_start, test_start) indices for a time-ordered split."""
    val_start = int(np.floor(T * (1.0 - (val_ratio + test_ratio))))
    test_start = int(np.floor(T * (1.0 - test_ratio)))
    return val_start, test_start

# ----------------------------
# Main build
# ----------------------------
def main():
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    
    csv_paths = sorted(glob.glob(os.path.join(config.DATA_PROCESSED_DIR, "asteroid_*.csv")))
    
    if not csv_paths:
        raise FileNotFoundError(
            f"No asteroid_*.csv found in {config.DATA_PROCESSED_DIR}. "
            f"Did you run the fetch-data.ipynb first?"
        )

    # Use first N asteroids as defined in config
    csv_paths = csv_paths[: config.N_ASTEROIDS]
    
    print(f"--- Processing {len(csv_paths)} asteroids ---")
    
    # These lists will hold the windows from ALL asteroids
    X_tr_list, Y_tr_list, ID_tr_list = [], [], []
    X_va_list, Y_va_list, ID_va_list = [], [], []
    X_te_list, Y_te_list, ID_te_list = [], [], []

    processed_asteroids = []

    # Loop through each asteroid file, assign it an ID
    for asteroid_id, path in enumerate(csv_paths):
        name = os.path.basename(path).split('_')[1]
        print(f"\n[Asteroid {asteroid_id}: {name}]")
        
        df = pd.read_csv(path)
        
        # 1. Validate and get feature array
        missing_cols = [f for f in config.FEATURES if f not in df.columns]
        if missing_cols:
            print(f"[skip] Missing required cols: {missing_cols}")
            continue
            
        feat_raw = df[config.FEATURES].values.astype(np.float64)
        
        if len(feat_raw) < (config.INPUT_STEPS + config.OUTPUT_STEPS + 20):
            print(f"[skip] Series too short: {len(feat_raw)} rows")
            continue
            
        # 2. Split by time (on raw array)
        val_start, test_start = time_split_indices(
            len(feat_raw), config.VAL_RATIO, config.TEST_RATIO
        )
        tr_raw = feat_raw[:val_start]
        va_raw = feat_raw[val_start:test_start]
        te_raw = feat_raw[test_start:]

        # 3. Create windows for each split
        Xtr, Ytr = create_sliding_windows(tr_raw, config.INPUT_STEPS, config.OUTPUT_STEPS)
        Xva, Yva = create_sliding_windows(va_raw, config.INPUT_STEPS, config.OUTPUT_STEPS)
        Xte, Yte = create_sliding_windows(te_raw, config.INPUT_STEPS, config.OUTPUT_STEPS)

        print(f"Windows created: Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)}")

        # 4. Create corresponding ID arrays
        # All windows from this file get the same asteroid_id
        if len(Xtr):
            X_tr_list.append(Xtr); Y_tr_list.append(Ytr)
            ID_tr_list.append(np.full(len(Xtr), asteroid_id, dtype=np.int32))
        if len(Xva):
            X_va_list.append(Xva); Y_va_list.append(Yva)
            ID_va_list.append(np.full(len(Xva), asteroid_id, dtype=np.int32))
        if len(Xte):
            X_te_list.append(Xte); Y_te_list.append(Yte)
            ID_te_list.append(np.full(len(Xte), asteroid_id, dtype=np.int32))
            
        processed_asteroids.append({"id": asteroid_id, "name": name})

    print("\n--- Stacking all datasets ---")

    # 5. Stack all arrays from all asteroids
    X_train = np.vstack(X_tr_list)
    Y_train = np.vstack(Y_tr_list)
    ID_train = np.concatenate(ID_tr_list)

    X_val = np.vstack(X_va_list)
    Y_val = np.vstack(Y_va_list)
    ID_val = np.concatenate(ID_va_list)

    X_test = np.vstack(X_te_list)
    Y_test = np.vstack(Y_te_list)
    ID_test = np.concatenate(ID_te_list)
    
    # 6. Save to .npz
    def save_split(name, X, Y, IDS):
        path = os.path.join(config.DATASET_DIR, f"{name}.npz")
        print(f"[save] {name}.npz | X: {X.shape} | Y: {Y.shape} | IDs: {IDS.shape}")
        np.savez_compressed(
            path,
            X=X,
            Y=Y,
            asteroid_ids=IDS,
            input_steps=config.INPUT_STEPS,
            output_steps=config.OUTPUT_STEPS,
            features=config.FEATURES
        )

    save_split("train", X_train, Y_train, ID_train)
    save_split("val", X_val, Y_val, ID_val)
    save_split("test", X_test, Y_test, ID_test)

    # 7. Save a manifest file
    manifest = {
        "project": "Foundational Asteroid Transformer",
        "features": config.FEATURES,
        "num_features": config.NUM_FEATURES,
        "asteroids": processed_asteroids,
        "num_asteroids": len(processed_asteroids),
        "input_steps": config.INPUT_STEPS,
        "output_steps": config.OUTPUT_STEPS,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(config.DATASET_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] manifest.json")
    
    print("\n--- Pre-processing complete! ---")

if __name__ == "__main__":
    main()
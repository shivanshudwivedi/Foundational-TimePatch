"""
Defines the custom PyTorch Dataset for the Asteroid Transformer.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os

# Import from our config file
import config

class AsteroidDataset(Dataset):
    """
    Custom PyTorch Dataset for loading asteroid trajectory windows.
    
    Reads data from a .npz file and provides tuples of:
    (x_window, y_window, asteroid_id)
    """
    
    def __init__(self, split: str = "train"):
        """
        Initializes the dataset by loading the .npz file.
        
        Args:
            split (str): One of "train", "val", or "test" to load the
                         corresponding .npz file.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split '{split}'. Must be 'train', 'val', or 'test'.")
            
        self.split = split
        npz_path = os.path.join(config.DATASET_DIR, f"{self.split}.npz")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Dataset file not found at {npz_path}. "
                f"Did you run pre-process.py first?"
            )
            
        print(f"[Dataset] Loading {self.split} data from {npz_path}...")
        
        try:
            # Load the pre-processed data
            with np.load(npz_path) as data:
                # X shape: (N, INPUT_STEPS, NUM_FEATURES)
                self.X = data['X']
                # Y shape: (N, OUTPUT_STEPS, NUM_FEATURES)
                self.Y = data['Y']
                # asteroid_ids shape: (N,)
                self.asteroid_ids = data['asteroid_ids']
                
            # Convert to PyTorch tensors for efficiency
            # We use float32 for model inputs and long for embedding indices
            self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
            self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32)
            self.ID_tensor = torch.tensor(self.asteroid_ids, dtype=torch.long)
            
            self.n_samples = self.X_tensor.shape[0]
            print(f"[Dataset] Loaded {self.n_samples} samples for {self.split} split.")
            
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            raise

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset by index.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            tuple: (x, y, asteroid_id)
                - x: (INPUT_STEPS, NUM_FEATURES)
                - y: (OUTPUT_STEPS, NUM_FEATURES)
                - asteroid_id: (scalar)
        """
        return self.X_tensor[idx], self.Y_tensor[idx], self.ID_tensor[idx]
"""
src/data.py

Defines the PyTorch Lightning LightningDataModule.
This module encapsulates all data loading and preparation.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import our existing dataset class
from src.dataset import AsteroidDataset
import config

class AsteroidDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size: int = config.BATCH_SIZE, num_workers: int = 4):
        super().__init__()
        # Save batch_size and num_workers as hyperparameters
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Called on every GPU to load datasets.
        'stage' can be 'fit', 'validate', 'test', or 'predict'.
        This prevents data from being loaded multiple times.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = AsteroidDataset(split="train")
            self.val_dataset = AsteroidDataset(split="val")
        
        if stage == 'test' or stage is None:
            self.test_dataset = AsteroidDataset(split="test")
            
    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        """Returns the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

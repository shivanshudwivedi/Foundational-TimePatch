"""
src/system.py

Defines the PyTorch Lightning LightningModule.
This is the core of our project, combining the model and the
training/validation/testing logic.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# Import our existing model architecture from src/model.py
from src.model import AsteroidTransformer
import config

class AsteroidForecastingSystem(pl.LightningModule):
    
    def __init__(self, lr: float = config.LR):
        super().__init__()
        # save_hyperparameters() logs all __init__ args (like lr)
        # to the checkpoint file and makes them accessible via self.hparams
        self.save_hyperparameters()
        
        # 1. Instantiate the model (from src/model.py)
        self.model = AsteroidTransformer()
        
        # 2. Define the loss function
        self.criterion = nn.MSELoss()
        
        # 3. Define metrics for logging
        # We use torchmetrics for correct logging in distributed settings
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_rmse = MeanSquaredError(squared=False) # Calculate RMSE
        self.test_mae = MeanAbsoluteError()

    def forward(self, x, asteroid_ids):
        """Forward pass is just the model's forward pass."""
        return self.model(x, asteroid_ids)

    def _shared_step(self, batch):
        """Helper function to avoid code duplication in train/val steps."""
        x, y, asteroid_ids = batch
        preds = self(x, asteroid_ids)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        # Log training loss and MSE
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_mse.update(preds, y)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        # Log validation loss and MSE
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mse.update(preds, y)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        
        # Update our dedicated test metrics
        self.test_rmse.update(preds, y)
        self.test_mae.update(preds, y)
        
        # Log test metrics
        self.log('test_rmse', self.test_rmse, on_step=False, on_epoch=True)
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Defines the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

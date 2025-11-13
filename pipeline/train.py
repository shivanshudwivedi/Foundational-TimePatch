"""
NEW train.py

This is the main training script, refactored for PyTorch Lightning.
It initializes the DataModule, the System, and the Trainer,
then calls 'trainer.fit()'.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from src.data import AsteroidDataModule
from src.system import AsteroidForecastingSystem

def main():
    print("--- Starting PyTorch Lightning Training ---")
    
    # 1. Initialize DataModule
    datamodule = AsteroidDataModule(batch_size=config.BATCH_SIZE)
    
    # 2. Initialize System (LightningModule)
    system = AsteroidForecastingSystem(lr=config.LR)
    
    # 3. Configure Callbacks
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss', # The metric to monitor
        patience=config.PATIENCE,
        verbose=True,
        mode='min' # 'min' means we want to minimize the metric
    )
    
    # Model checkpointing (saves the best model)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.ARTIFACT_DIR,
        filename='best_model', # Saves to 'best_model.ckpt'
        monitor='val_loss',
        save_top_k=1, # Save only the best model
        mode='min',
        verbose=True
    )
    
    # 4. Configure Logger (e.g., TensorBoard)
    logger = TensorBoardLogger(save_dir=config.LOG_DIR, name="asteroid_transformer")
    
    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="auto", # Automatically selects (CPU, GPU, MPS)
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10
    )
    
    # 6. Start Training!
    print("Beginning training...")
    trainer.fit(system, datamodule)
    
    print("--- Training Complete ---")
    print(f"Best model checkpoint saved to: {checkpoint_callback.best_model_path}")
    
    # 7. Optional: Run test immediately after training
    print("Running final test set evaluation...")
    # 'ckpt_path="best"' automatically loads the best model saved
    test_results = trainer.test(datamodule=datamodule, ckpt_path='best')
    print("Test Results:", test_results)

if __name__ == "__main__":
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    main()

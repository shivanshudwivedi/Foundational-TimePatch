"""
NEW evaluate.py

This script loads the best-trained model from its checkpoint
and runs the evaluation on the test set.
"""

import os
import pytorch_lightning as pl
import json
import matplotlib.pyplot as plt
import torch

import config
from src.data import AsteroidDataModule
from src.system import AsteroidForecastingSystem

# A (small) index from the test set to plot
SAMPLE_INDEX = 0

def main():
    print("--- Starting PyTorch Lightning Evaluation ---")
    
    # 1. Check for the best checkpoint
    if not os.path.exists(config.BEST_CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {config.BEST_CHECKPOINT_PATH}")
        print("Please run train.py first.")
        return

    print(f"Loading model from {config.BEST_CHECKPOINT_PATH}")

    # 2. Initialize DataModule
    datamodule = AsteroidDataModule(batch_size=config.BATCH_SIZE)
    
    # 3. Load the system from the checkpoint
    system = AsteroidForecastingSystem.load_from_checkpoint(config.BEST_CHECKPOINT_PATH)
    system.to(config.DEVICE)
    system.eval()
    
    # 4. Initialize Trainer
    trainer = pl.Trainer(accelerator="auto")
    
    # 5. Run Test
    # This will run system.test_step() on all test data
    print("Running evaluation on test set...")
    results = trainer.test(system, datamodule)
    
    # The results are a list of dictionaries
    final_metrics = results[0]
    print("\n--- Evaluation Results ---")
    print(f"  Test RMSE: {final_metrics['test_rmse']:.8f}")
    print(f"  Test MAE:  {final_metrics['test_mae']:.8f}")
    
    # 6. Compare to Baseline
    try:
        with open(config.BASELINE_REPORT, 'r') as f:
            baseline_report = json.load(f)
        baseline_rmse = baseline_report.get('test_rmse_state_only', 
                                          baseline_report.get('test_rmse'))
        
        print("\n--- Performance Comparison ---")
        print(f"  FNN Baseline RMSE:   {baseline_rmse:.8f}")
        print(f"  Transformer RMSE:    {final_metrics['test_rmse']:.8f}")
        
        improvement = (baseline_rmse - final_metrics['test_rmse']) / baseline_rmse * 100
        print(f"  Improvement:         {improvement:.2f}%")
        
    except FileNotFoundError:
        print("\nBaseline report not found. Skipping comparison.")
    except Exception as e:
        print(f"\nCould not read baseline report: {e}")

    # 7. Save a simple report
    report = {
        "model": "AsteroidTransformer (Lightning)",
        "checkpoint_path": config.BEST_CHECKPOINT_PATH,
        **final_metrics,
        "hparams": dict(system.hparams)
    }
    with open(config.TRANSFORMER_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {config.TRANSFORMER_REPORT}")

    # 8. Plot a sample forecast
    print("Generating sample forecast plot...")
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    # Get one batch
    x, y, asteroid_ids = next(iter(test_loader))
    
    # Get one sample from the batch
    x_sample = x[SAMPLE_INDEX].unsqueeze(0).to(config.DEVICE)
    y_sample = y[SAMPLE_INDEX].numpy()
    id_sample = asteroid_ids[SAMPLE_INDEX].unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        pred_sample = system(x_sample, id_sample).cpu().numpy()[0]
        
    true_seq = y_sample # Shape: (OUTPUT_STEPS, 6)
    pred_seq = pred_sample # Shape: (OUTPUT_STEPS, 6)
    
    plt.figure(figsize=(8, 5))
    plt.plot(true_seq[:, 0], true_seq[:, 1], 'o-', label="True (XY)")
    plt.plot(pred_seq[:, 0], pred_seq[:, 1], 'x--', label="Predicted (XY)")
    plt.xlabel("X [AU]")
    plt.ylabel("Y [AU]")
    plt.title(f"Transformer Sample Forecast (idx {SAMPLE_INDEX}) - XY Plane")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(config.TRANSFORMER_PLOT, dpi=150)
    print(f"Saved sample forecast plot to {config.TRANSFORMER_PLOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main training script for beat tracking implementation."""

import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticBeatDataset
from src.models.baseline import BaselineBeatTracker
from src.models.advanced import RNNBeatTracker
from src.train.trainer import Trainer
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set up logging
    setup_logging(
        level=cfg.logging.level,
        log_file=Path(cfg.paths.logs_dir) / "training.log",
    )
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(cfg.paths.output_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved configuration to {config_path}")
    
    # Create datasets
    logger.info("Creating synthetic datasets...")
    
    train_dataset = SyntheticBeatDataset(
        num_samples=cfg.data.generation.num_samples,
        duration_range=tuple(cfg.data.generation.duration_range),
        tempo_range=tuple(cfg.data.generation.tempo_range),
        time_signature=tuple(cfg.data.generation.time_signature),
        sample_rate=cfg.data.generation.sample_rate,
        split="train",
        data_dir=cfg.paths.data_dir,
    )
    
    val_dataset = SyntheticBeatDataset(
        num_samples=cfg.data.generation.num_samples // 4,
        duration_range=tuple(cfg.data.generation.duration_range),
        tempo_range=tuple(cfg.data.generation.tempo_range),
        time_signature=tuple(cfg.data.generation.time_signature),
        sample_rate=cfg.data.generation.sample_rate,
        split="val",
        data_dir=cfg.paths.data_dir,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.loader.batch_size,
        shuffle=cfg.data.loader.shuffle,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.loader.batch_size,
        shuffle=False,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory,
    )
    
    logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # Create model
    logger.info(f"Creating {cfg.model.name} model...")
    
    if cfg.model.name == "baseline":
        model = BaselineBeatTracker(
            sample_rate=cfg.model.audio.sample_rate,
            hop_length=cfg.model.audio.hop_length,
            n_fft=cfg.model.audio.n_fft,
            n_mels=cfg.model.audio.n_mels,
            tempo_min=cfg.model.beat_tracking.tempo_min,
            tempo_max=cfg.model.beat_tracking.tempo_max,
            units=cfg.model.beat_tracking.units,
            tightness=cfg.model.beat_tracking.tightness,
            trim=cfg.model.beat_tracking.trim,
            aggregate=cfg.model.beat_tracking.aggregate,
        )
        
        # For baseline model, we'll use a simple evaluation approach
        logger.info("Baseline model created. Running evaluation...")
        evaluate_baseline_model(model, val_dataset, output_dir)
        return
    
    elif cfg.model.name == "rnn_beat_tracker":
        model = RNNBeatTracker(
            input_dim=cfg.model.architecture.input_dim,
            hidden_dim=cfg.model.architecture.hidden_dim,
            num_layers=cfg.model.architecture.num_layers,
            dropout=cfg.model.architecture.dropout,
            bidirectional=cfg.model.architecture.bidirectional,
            rnn_type=cfg.model.architecture.rnn_type,
            sample_rate=cfg.model.audio.sample_rate,
            hop_length=cfg.model.audio.hop_length,
            n_fft=cfg.model.audio.n_fft,
            n_mels=cfg.model.audio.n_mels,
            tempo_min=cfg.model.beat_tracking.tempo_min,
            tempo_max=cfg.model.beat_tracking.tempo_max,
            units=cfg.model.beat_tracking.units,
            tightness=cfg.model.beat_tracking.tightness,
            trim=cfg.model.beat_tracking.trim,
        )
    
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.training.scheduler.mode,
        factor=cfg.training.scheduler.factor,
        patience=cfg.training.scheduler.patience,
        verbose=cfg.training.scheduler.verbose,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_interval=cfg.training.logging.log_every_n_steps,
    )
    
    # Train model
    logger.info("Starting training...")
    
    history = trainer.train(
        num_epochs=cfg.training.epochs,
        save_best=True,
        early_stopping_patience=cfg.training.early_stopping_patience,
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    import json
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. History saved to {history_path}")
    
    # Save final model
    final_model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


def evaluate_baseline_model(
    model: BaselineBeatTracker,
    dataset: SyntheticBeatDataset,
    output_dir: Path,
) -> None:
    """Evaluate baseline model on synthetic dataset.
    
    Args:
        model: Baseline beat tracker model.
        dataset: Evaluation dataset.
        output_dir: Output directory for results.
    """
    logger.info("Evaluating baseline model...")
    
    results = []
    
    for i in range(min(10, len(dataset))):  # Evaluate on first 10 samples
        sample = dataset[i]
        
        # Get predictions
        tempo_pred, beats_pred = model.predict(sample["audio"])
        
        # Ground truth
        tempo_gt = sample["tempo"]
        beats_gt = sample["beat_times"]
        
        # Calculate metrics
        tempo_error = abs(tempo_pred - tempo_gt) / tempo_gt
        
        results.append({
            "sample_id": sample["sample_id"],
            "tempo_gt": tempo_gt,
            "tempo_pred": tempo_pred,
            "tempo_error": tempo_error,
            "num_beats_gt": len(beats_gt),
            "num_beats_pred": len(beats_pred),
        })
    
    # Calculate average metrics
    avg_tempo_error = sum(r["tempo_error"] for r in results) / len(results)
    
    logger.info(f"Baseline model evaluation:")
    logger.info(f"  Average tempo error: {avg_tempo_error:.3f}")
    logger.info(f"  Evaluated on {len(results)} samples")
    
    # Save results
    import json
    results_path = output_dir / "baseline_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()

"""Training utilities for beat tracking models."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics.beat_tracking import BeatTrackingMetrics
from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BeatTrackingLoss(nn.Module):
    """Loss function for beat tracking training."""
    
    def __init__(
        self,
        tempo_weight: float = 0.3,
        beat_weight: float = 0.7,
        continuity_weight: float = 0.2,
    ):
        """Initialize beat tracking loss.
        
        Args:
            tempo_weight: Weight for tempo prediction loss.
            beat_weight: Weight for beat prediction loss.
            continuity_weight: Weight for continuity loss.
        """
        super().__init__()
        self.tempo_weight = tempo_weight
        self.beat_weight = beat_weight
        self.continuity_weight = continuity_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculate loss.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dictionary containing individual and total losses.
        """
        losses = {}
        
        # Tempo prediction loss
        if "tempo_pred" in predictions and "tempo_target" in targets:
            tempo_loss = self.mse_loss(predictions["tempo_pred"], targets["tempo_target"])
            losses["tempo_loss"] = tempo_loss
        
        # Beat prediction loss
        if "beat_pred" in predictions and "beat_target" in targets:
            beat_loss = self.bce_loss(predictions["beat_pred"], targets["beat_target"])
            losses["beat_loss"] = beat_loss
        
        # Total loss
        total_loss = (
            self.tempo_weight * losses.get("tempo_loss", 0) +
            self.beat_weight * losses.get("beat_loss", 0)
        )
        
        losses["total_loss"] = total_loss
        
        return losses


class Trainer:
    """Trainer for beat tracking models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        log_interval: int = 50,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            loss_fn: Loss function.
            device: Device to use for training.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Logging interval.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or BeatTrackingLoss()
        self.device = get_device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.log_interval = log_interval
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics
        self.metrics = BeatTrackingMetrics()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Initialized Trainer with device={self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model predictions
            if hasattr(self.model, 'forward'):
                predictions = self.model(batch["mel_spec"])
            else:
                # For baseline model
                predictions = self.model(batch["audio"])
            
            # Calculate loss
            losses = self.loss_fn(predictions, batch)
            loss = losses["total_loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}"
                )
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        return {
            "train_loss": avg_loss,
            "num_batches": num_batches,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    predictions = self.model(batch["mel_spec"])
                else:
                    # For baseline model
                    predictions = self.model(batch["audio"])
                
                # Calculate loss
                losses = self.loss_fn(predictions, batch)
                loss = losses["total_loss"]
                
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions)
                all_targets.append(batch)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate beat tracking metrics
        beat_metrics = self._calculate_beat_metrics(all_predictions, all_targets)
        
        return {
            "val_loss": avg_loss,
            **beat_metrics,
        }
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            save_best: Whether to save the best model.
            early_stopping_patience: Early stopping patience.
            
        Returns:
            Training history.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f_measure": [],
            "val_continuity": [],
            "val_accuracy": [],
        }
        
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Update history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_f_measure"].append(val_metrics.get("f_measure", 0.0))
            history["val_continuity"].append(val_metrics.get("continuity", 0.0))
            history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val F-Measure: {val_metrics.get('f_measure', 0.0):.4f}"
            )
            
            # Save checkpoint
            if save_best and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                best_epoch = epoch
                self.save_checkpoint("best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if (early_stopping_patience and 
                patience_counter >= early_stopping_patience):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary.
            
        Returns:
            Batch moved to device.
        """
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _calculate_beat_metrics(
        self,
        predictions: list,
        targets: list,
    ) -> Dict[str, float]:
        """Calculate beat tracking metrics.
        
        Args:
            predictions: List of model predictions.
            targets: List of ground truth targets.
            
        Returns:
            Dictionary containing beat tracking metrics.
        """
        # This is a simplified implementation
        # In practice, you would need to convert predictions to beat times
        # and calculate metrics using the BeatTrackingMetrics class
        
        return {
            "f_measure": 0.0,
            "continuity": 0.0,
            "accuracy": 0.0,
        }

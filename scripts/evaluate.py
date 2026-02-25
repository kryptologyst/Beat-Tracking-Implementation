#!/usr/bin/env python3
"""Evaluation script for beat tracking implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.synthetic import SyntheticBeatDataset
from src.models.baseline import BaselineBeatTracker
from src.models.advanced import RNNBeatTracker
from src.metrics.beat_tracking import BeatTrackingMetrics
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set up logging
    setup_logging(
        level=cfg.logging.level,
        log_file=Path(cfg.paths.logs_dir) / "evaluation.log",
    )
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test dataset
    logger.info("Creating test dataset...")
    
    test_dataset = SyntheticBeatDataset(
        num_samples=100,  # Smaller test set
        duration_range=tuple(cfg.data.generation.duration_range),
        tempo_range=tuple(cfg.data.generation.tempo_range),
        time_signature=tuple(cfg.data.generation.time_signature),
        sample_rate=cfg.data.generation.sample_rate,
        split="test",
        data_dir=cfg.paths.data_dir,
    )
    
    logger.info(f"Created test dataset with {len(test_dataset)} samples")
    
    # Initialize metrics
    metrics = BeatTrackingMetrics()
    
    # Evaluate models
    results = {}
    
    # Baseline model
    logger.info("Evaluating baseline model...")
    baseline_results = evaluate_model(
        model_name="baseline",
        model_config=cfg.model,
        dataset=test_dataset,
        metrics=metrics,
        device=device,
    )
    results["baseline"] = baseline_results
    
    # RNN model (if available)
    rnn_model_path = output_dir / "final_model.pth"
    if rnn_model_path.exists():
        logger.info("Evaluating RNN model...")
        rnn_results = evaluate_model(
            model_name="rnn",
            model_config=cfg.model,
            dataset=test_dataset,
            metrics=metrics,
            device=device,
            checkpoint_path=rnn_model_path,
        )
        results["rnn"] = rnn_results
    else:
        logger.warning("RNN model checkpoint not found, skipping RNN evaluation")
    
    # Create leaderboard
    leaderboard = create_leaderboard(results)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    
    # Print results
    print("\n" + "="*60)
    print("BEAT TRACKING EVALUATION RESULTS")
    print("="*60)
    
    print("\nLEADERBOARD:")
    print(leaderboard.to_string(index=False))
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Leaderboard saved to: {leaderboard_path}")
    
    logger.info("Evaluation completed successfully")


def evaluate_model(
    model_name: str,
    model_config: DictConfig,
    dataset: SyntheticBeatDataset,
    metrics: BeatTrackingMetrics,
    device: torch.device,
    checkpoint_path: Path = None,
) -> Dict[str, Any]:
    """Evaluate a single model.
    
    Args:
        model_name: Name of the model to evaluate.
        model_config: Model configuration.
        dataset: Test dataset.
        metrics: Beat tracking metrics calculator.
        device: Device to use.
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        Dictionary containing evaluation results.
    """
    # Create model
    if model_name == "baseline":
        model = BaselineBeatTracker(
            sample_rate=model_config.audio.sample_rate,
            hop_length=model_config.audio.hop_length,
            n_fft=model_config.audio.n_fft,
            n_mels=model_config.audio.n_mels,
            tempo_min=model_config.beat_tracking.tempo_min,
            tempo_max=model_config.beat_tracking.tempo_max,
            units=model_config.beat_tracking.units,
            tightness=model_config.beat_tracking.tightness,
            trim=model_config.beat_tracking.trim,
            aggregate=model_config.beat_tracking.aggregate,
        )
        
        # Load checkpoint if available
        if checkpoint_path and checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    elif model_name == "rnn":
        model = RNNBeatTracker(
            input_dim=model_config.architecture.input_dim,
            hidden_dim=model_config.architecture.hidden_dim,
            num_layers=model_config.architecture.num_layers,
            dropout=model_config.architecture.dropout,
            bidirectional=model_config.architecture.bidirectional,
            rnn_type=model_config.architecture.rnn_type,
            sample_rate=model_config.audio.sample_rate,
            hop_length=model_config.audio.hop_length,
            n_fft=model_config.audio.n_fft,
            n_mels=model_config.audio.n_mels,
            tempo_min=model_config.beat_tracking.tempo_min,
            tempo_max=model_config.beat_tracking.tempo_max,
            units=model_config.beat_tracking.units,
            tightness=model_config.beat_tracking.tightness,
            trim=model_config.beat_tracking.trim,
        )
        
        # Load checkpoint
        if checkpoint_path and checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            logger.warning("No checkpoint found for RNN model, using random weights")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    
    # Evaluation results
    all_metrics = []
    tempo_errors = []
    
    logger.info(f"Evaluating {model_name} on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            try:
                # Get predictions
                if model_name == "baseline":
                    tempo_pred, beats_pred = model.predict(sample["audio"])
                else:  # RNN
                    result = model.predict_beats(sample["audio"])
                    tempo_pred = result["tempo"]
                    beats_pred = result["beats"]
                
                # Ground truth
                tempo_gt = sample["tempo"]
                beats_gt = sample["beat_times"]
                
                # Calculate beat tracking metrics
                beat_metrics = metrics.evaluate(beats_pred, beats_gt)
                
                # Calculate tempo accuracy
                tempo_accuracy = metrics.calculate_tempo_accuracy(tempo_pred, tempo_gt)
                
                # Store results
                sample_results = {
                    "sample_id": sample["sample_id"],
                    "tempo_gt": tempo_gt,
                    "tempo_pred": tempo_pred,
                    "tempo_accuracy": tempo_accuracy,
                    "tempo_error": abs(tempo_pred - tempo_gt) / tempo_gt,
                    "num_beats_gt": len(beats_gt),
                    "num_beats_pred": len(beats_pred),
                    **beat_metrics,
                }
                
                all_metrics.append(sample_results)
                tempo_errors.append(sample_results["tempo_error"])
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
    
    # Calculate aggregate metrics
    if all_metrics:
        aggregate_metrics = {
            "num_samples": len(all_metrics),
            "tempo_accuracy": np.mean([m["tempo_accuracy"] for m in all_metrics]),
            "tempo_error_mean": np.mean(tempo_errors),
            "tempo_error_std": np.std(tempo_errors),
            "f_measure_mean": np.mean([m["f_measure"] for m in all_metrics]),
            "f_measure_std": np.std([m["f_measure"] for m in all_metrics]),
            "continuity_mean": np.mean([m["continuity"] for m in all_metrics]),
            "continuity_std": np.std([m["continuity"] for m in all_metrics]),
            "accuracy_mean": np.mean([m["accuracy"] for m in all_metrics]),
            "accuracy_std": np.std([m["accuracy"] for m in all_metrics]),
            "cmlc_mean": np.mean([m["cmlc"] for m in all_metrics]),
            "cmlc_std": np.std([m["cmlc"] for m in all_metrics]),
            "cmlt_mean": np.mean([m["cmlt"] for m in all_metrics]),
            "cmlt_std": np.std([m["cmlt"] for m in all_metrics]),
            "amlc_mean": np.mean([m["amlc"] for m in all_metrics]),
            "amlc_std": np.std([m["amlc"] for m in all_metrics]),
            "amlt_mean": np.mean([m["amlt"] for m in all_metrics]),
            "amlt_std": np.std([m["amlt"] for m in all_metrics]),
        }
    else:
        aggregate_metrics = {"num_samples": 0}
    
    return {
        "model_name": model_name,
        "model_info": model.get_model_info(),
        "aggregate_metrics": aggregate_metrics,
        "sample_results": all_metrics,
    }


def create_leaderboard(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a leaderboard from evaluation results.
    
    Args:
        results: Dictionary containing evaluation results for each model.
        
    Returns:
        DataFrame containing the leaderboard.
    """
    leaderboard_data = []
    
    for model_name, model_results in results.items():
        metrics = model_results["aggregate_metrics"]
        
        if metrics["num_samples"] > 0:
            leaderboard_data.append({
                "Model": model_name,
                "Samples": metrics["num_samples"],
                "Tempo Accuracy": f"{metrics['tempo_accuracy']:.3f}",
                "F-Measure": f"{metrics['f_measure_mean']:.3f} ± {metrics['f_measure_std']:.3f}",
                "Continuity": f"{metrics['continuity_mean']:.3f} ± {metrics['continuity_std']:.3f}",
                "Accuracy": f"{metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}",
                "CMLC": f"{metrics['cmlc_mean']:.3f} ± {metrics['cmlc_std']:.3f}",
                "CMLT": f"{metrics['cmlt_mean']:.3f} ± {metrics['cmlt_std']:.3f}",
                "AMLC": f"{metrics['amlc_mean']:.3f} ± {metrics['amlc_std']:.3f}",
                "AMLT": f"{metrics['amlt_mean']:.3f} ± {metrics['amlt_std']:.3f}",
            })
    
    return pd.DataFrame(leaderboard_data)


if __name__ == "__main__":
    main()

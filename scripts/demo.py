#!/usr/bin/env python3
"""Simple demo script for beat tracking."""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.models.baseline import BaselineBeatTracker
from src.utils.audio import extract_mel_spectrogram, extract_onset_strength
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def create_sample_audio(duration: float = 10.0, sample_rate: int = 22050) -> np.ndarray:
    """Create a sample audio with known beat pattern.
    
    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate.
        
    Returns:
        Generated audio array.
    """
    logger.info(f"Creating sample audio with duration {duration}s")
    
    # Create time axis
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Add kick drum on every beat (120 BPM)
    tempo = 120  # BPM
    beat_interval = 60.0 / tempo  # seconds per beat
    
    for i in range(int(duration / beat_interval)):
        beat_time = i * beat_interval
        beat_samples = int(beat_time * sample_rate)
        
        if beat_samples < len(audio):
            # Simple kick drum sound
            kick_duration = 0.1
            kick_samples = int(kick_duration * sample_rate)
            kick_end = min(beat_samples + kick_samples, len(audio))
            
            # Generate kick sound
            kick_t = t[beat_samples:kick_end] - beat_time
            kick_sound = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t * 20)
            audio[beat_samples:kick_end] += kick_sound
    
    # Add some hi-hat
    for i in range(int(duration / beat_interval)):
        beat_time = i * beat_interval
        beat_samples = int(beat_time * sample_rate)
        
        if beat_samples < len(audio):
            # Hi-hat on every beat
            hihat_duration = 0.02
            hihat_samples = int(hihat_duration * sample_rate)
            hihat_end = min(beat_samples + hihat_samples, len(audio))
            
            # Generate hi-hat sound (noise)
            hihat_sound = np.random.normal(0, 0.3, hihat_end - beat_samples)
            audio[beat_samples:hihat_end] += hihat_sound
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio


def visualize_results(
    audio: np.ndarray,
    beats: np.ndarray,
    onset_env: np.ndarray,
    tempo: float,
    sample_rate: int,
    output_path: Path,
) -> None:
    """Visualize beat tracking results.
    
    Args:
        audio: Input audio array.
        beats: Detected beat times.
        onset_env: Onset strength envelope.
        tempo: Estimated tempo.
        sample_rate: Sample rate.
        output_path: Output path for visualization.
    """
    logger.info("Creating visualization")
    
    # Create time axis
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(audio))
    onset_time = np.linspace(0, duration, len(onset_env))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot audio waveform
    axes[0].plot(time_axis, audio, 'b-', linewidth=0.5)
    axes[0].set_title(f"Audio Waveform (Estimated Tempo: {tempo:.1f} BPM)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Plot onset envelope
    axes[1].plot(onset_time, onset_env, 'g-', linewidth=1)
    axes[1].set_title("Onset Strength Envelope")
    axes[1].set_ylabel("Onset Strength")
    axes[1].grid(True, alpha=0.3)
    
    # Plot detected beats
    axes[2].plot(onset_time, onset_env, 'g-', linewidth=1, alpha=0.7, label="Onset Envelope")
    
    if len(beats) > 0:
        beat_y = np.ones(len(beats)) * np.max(onset_env)
        axes[2].scatter(beats, beat_y, color='red', s=50, marker='^', 
                      label=f"Detected Beats ({len(beats)})", zorder=5)
    
    axes[2].set_title("Beat Detection Results")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Onset Strength")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")


def main():
    """Main demo function."""
    logger.info("Starting beat tracking demo")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device info
    device = get_device("auto")
    logger.info(f"Using device: {device}")
    
    # Create sample audio
    sample_rate = 22050
    duration = 15.0  # seconds
    audio = create_sample_audio(duration=duration, sample_rate=sample_rate)
    
    logger.info(f"Generated audio: {len(audio)} samples, {duration}s duration")
    
    # Create beat tracker
    model = BaselineBeatTracker(
        sample_rate=sample_rate,
        tempo_min=60,
        tempo_max=200,
    )
    
    logger.info("Created baseline beat tracker")
    
    # Perform beat tracking
    logger.info("Performing beat tracking...")
    result = model.forward(audio)
    
    tempo = result["tempo"]
    beats = result["beats"]
    onset_env = result["onset_envelope"]
    
    logger.info(f"Beat tracking completed:")
    logger.info(f"  Estimated tempo: {tempo:.1f} BPM")
    logger.info(f"  Detected beats: {len(beats)}")
    
    if len(beats) > 0:
        beat_interval = np.mean(np.diff(beats))
        logger.info(f"  Average beat interval: {beat_interval:.3f}s")
    
    # Create output directory
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize results
    viz_path = output_dir / "beat_tracking_demo.png"
    visualize_results(
        audio=audio,
        beats=beats,
        onset_env=onset_env,
        tempo=tempo,
        sample_rate=sample_rate,
        output_path=viz_path,
    )
    
    # Save beat times
    beats_path = output_dir / "detected_beats.txt"
    np.savetxt(beats_path, beats, fmt="%.6f")
    logger.info(f"Beat times saved to {beats_path}")
    
    # Print beat times
    print("\n" + "="*50)
    print("BEAT TRACKING DEMO RESULTS")
    print("="*50)
    print(f"Estimated Tempo: {tempo:.1f} BPM")
    print(f"Number of Beats: {len(beats)}")
    print(f"Audio Duration: {duration:.1f} seconds")
    
    if len(beats) > 0:
        print(f"Average Beat Interval: {beat_interval:.3f} seconds")
        print(f"Expected Beat Interval: {60.0/120:.3f} seconds")
        
        print("\nFirst 10 Beat Times:")
        for i, beat_time in enumerate(beats[:10]):
            print(f"  Beat {i+1}: {beat_time:.3f}s")
    
    print(f"\nVisualization saved to: {viz_path}")
    print(f"Beat times saved to: {beats_path}")
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()

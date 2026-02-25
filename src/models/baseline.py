"""Baseline beat tracking implementation using librosa."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

from ..utils.audio import extract_onset_strength, frames_to_time

logger = logging.getLogger(__name__)


class BaselineBeatTracker(nn.Module):
    """Baseline beat tracker using librosa's beat tracking algorithm.
    
    This implementation provides a simple but effective baseline for beat tracking
    using onset detection and tempo estimation from librosa.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        tempo_min: int = 60,
        tempo_max: int = 200,
        units: str = "time",
        tightness: float = 400.0,
        trim: bool = True,
        aggregate: str = "median",
    ):
        """Initialize baseline beat tracker.
        
        Args:
            sample_rate: Sample rate of input audio.
            hop_length: Hop length for analysis.
            n_fft: FFT window size.
            n_mels: Number of mel bins.
            tempo_min: Minimum tempo in BPM.
            tempo_max: Maximum tempo in BPM.
            units: Output units ('time' or 'frames').
            tightness: Beat tracking tightness parameter.
            trim: Whether to trim beats.
            aggregate: Aggregation method for onset strength.
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.units = units
        self.tightness = tightness
        self.trim = trim
        self.aggregate = aggregate
        
        logger.info(f"Initialized BaselineBeatTracker with sample_rate={sample_rate}")
    
    def forward(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """Perform beat tracking on input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            Dictionary containing tempo, beats, and onset envelope.
        """
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Extract onset strength envelope
        onset_env = extract_onset_strength(
            audio=audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            aggregate=self.aggregate,
        )
        
        # Perform beat tracking
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            start_bpm=self.tempo_min,
            tightness=self.tightness,
            trim=self.trim,
            units=self.units,
        )
        
        # Convert beats to time if needed
        if self.units == "frames":
            beats_time = frames_to_time(
                frames=beats,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        else:
            beats_time = beats
        
        return {
            "tempo": float(tempo),
            "beats": beats_time,
            "beats_frames": beats,
            "onset_envelope": onset_env,
        }
    
    def predict(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[float, np.ndarray]:
        """Predict beats for input audio (simplified interface).
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            Tuple of (tempo, beats_in_seconds).
        """
        result = self.forward(audio)
        return result["tempo"], result["beats"]
    
    def get_model_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get model information.
        
        Returns:
            Dictionary containing model parameters.
        """
        return {
            "name": "BaselineBeatTracker",
            "description": "Librosa-based baseline beat tracker",
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "tempo_min": self.tempo_min,
            "tempo_max": self.tempo_max,
            "units": self.units,
            "tightness": self.tightness,
            "trim": self.trim,
            "aggregate": self.aggregate,
        }

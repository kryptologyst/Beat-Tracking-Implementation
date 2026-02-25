"""Advanced RNN-based beat tracking implementation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.audio import extract_mel_spectrogram, frames_to_time

logger = logging.getLogger(__name__)


class RNNBeatTracker(nn.Module):
    """RNN-based beat tracker with temporal modeling.
    
    This implementation uses RNN layers to model temporal dependencies
    in onset strength patterns for improved beat tracking.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 128,
        tempo_min: int = 60,
        tempo_max: int = 200,
        units: str = "time",
        tightness: float = 400.0,
        trim: bool = True,
    ):
        """Initialize RNN beat tracker.
        
        Args:
            input_dim: Input feature dimension (mel bins).
            hidden_dim: Hidden dimension of RNN.
            num_layers: Number of RNN layers.
            dropout: Dropout rate.
            bidirectional: Whether to use bidirectional RNN.
            rnn_type: Type of RNN ('LSTM', 'GRU', 'RNN').
            sample_rate: Sample rate of input audio.
            hop_length: Hop length for analysis.
            n_fft: FFT window size.
            n_mels: Number of mel bins.
            tempo_min: Minimum tempo in BPM.
            tempo_max: Maximum tempo in BPM.
            units: Output units ('time' or 'frames').
            tightness: Beat tracking tightness parameter.
            trim: Whether to trim beats.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.units = units
        self.tightness = tightness
        self.trim = trim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # RNN layers
        rnn_class = getattr(nn, rnn_type)
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output layers
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.tempo_head = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output tempo probability
        )
        
        self.beat_head = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output beat probability
        )
        
        logger.info(f"Initialized RNNBeatTracker with {rnn_type} layers")
    
    def forward(
        self,
        mel_spec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            mel_spec: Mel spectrogram tensor of shape (batch, time, mel_bins).
            
        Returns:
            Dictionary containing tempo and beat predictions.
        """
        batch_size, seq_len, _ = mel_spec.shape
        
        # Input projection
        x = self.input_proj(mel_spec)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Predictions
        tempo_pred = self.tempo_head(rnn_out)  # (batch, time, 1)
        beat_pred = self.beat_head(rnn_out)    # (batch, time, 1)
        
        return {
            "tempo_pred": tempo_pred.squeeze(-1),  # (batch, time)
            "beat_pred": beat_pred.squeeze(-1),   # (batch, time)
        }
    
    def predict_beats(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Predict beats from audio using the trained model.
        
        Args:
            audio: Input audio array or tensor.
            threshold: Beat detection threshold.
            
        Returns:
            Dictionary containing tempo, beats, and predictions.
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(
            audio=audio.numpy(),
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )
        
        # Convert to tensor and add batch dimension
        mel_spec = torch.from_numpy(mel_spec).float().T.unsqueeze(0)  # (1, time, mel_bins)
        
        # Move to same device as model
        if mel_spec.device != next(self.parameters()).device:
            mel_spec = mel_spec.to(next(self.parameters()).device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(mel_spec)
            tempo_pred = outputs["tempo_pred"].squeeze(0)  # (time,)
            beat_pred = outputs["beat_pred"].squeeze(0)    # (time,)
        
        # Convert to numpy
        tempo_pred = tempo_pred.cpu().numpy()
        beat_pred = beat_pred.cpu().numpy()
        
        # Estimate tempo from tempo predictions
        tempo_values = np.linspace(self.tempo_min, self.tempo_max, len(tempo_pred))
        estimated_tempo = np.average(tempo_values, weights=tempo_pred)
        
        # Find beat locations
        beat_frames = np.where(beat_pred > threshold)[0]
        
        # Convert to time if needed
        if self.units == "time":
            beats = frames_to_time(
                frames=beat_frames,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        else:
            beats = beat_frames
        
        return {
            "tempo": float(estimated_tempo),
            "beats": beats,
            "beats_frames": beat_frames,
            "tempo_pred": tempo_pred,
            "beat_pred": beat_pred,
        }
    
    def get_model_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get model information.
        
        Returns:
            Dictionary containing model parameters.
        """
        return {
            "name": "RNNBeatTracker",
            "description": f"{self.rnn_type}-based beat tracker with temporal modeling",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "tempo_min": self.tempo_min,
            "tempo_max": self.tempo_max,
            "units": self.units,
            "tightness": self.tightness,
            "trim": self.trim,
        }

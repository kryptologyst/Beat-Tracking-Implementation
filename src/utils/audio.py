"""Audio processing utilities for beat tracking."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(
    file_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate. If None, use original.
        mono: Convert to mono if True.
        normalize: Normalize audio if True.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        audio, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=mono,
            dtype=np.float32,
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
        
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    format: str = "WAV",
) -> None:
    """Save audio array to file.
    
    Args:
        audio: Audio array to save.
        file_path: Output file path.
        sample_rate: Sample rate of audio.
        format: Audio format (WAV, FLAC, etc.).
    """
    try:
        sf.write(
            str(file_path),
            audio,
            sample_rate,
            format=format,
        )
        logger.info(f"Saved audio to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
        
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Extract mel spectrogram from audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bins.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        
    Returns:
        Mel spectrogram array.
    """
    if fmax is None:
        fmax = sample_rate // 2
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_onset_strength(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    aggregate: str = "median",
) -> np.ndarray:
    """Extract onset strength envelope.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        hop_length: Hop length for analysis.
        aggregate: Aggregation method for onset strength.
        
    Returns:
        Onset strength envelope.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=aggregate,
    )
    
    return onset_env


def frames_to_time(
    frames: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Convert frame indices to time in seconds.
    
    Args:
        frames: Frame indices.
        sample_rate: Sample rate of audio.
        hop_length: Hop length used in analysis.
        
    Returns:
        Time values in seconds.
    """
    return librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)


def time_to_frames(
    times: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Convert time in seconds to frame indices.
    
    Args:
        times: Time values in seconds.
        sample_rate: Sample rate of audio.
        hop_length: Hop length used in analysis.
        
    Returns:
        Frame indices.
    """
    return librosa.time_to_frames(times, sr=sample_rate, hop_length=hop_length)

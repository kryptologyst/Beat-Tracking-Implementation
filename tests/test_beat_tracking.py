"""Tests for beat tracking implementation."""

import numpy as np
import pytest
import torch

from src.models.baseline import BaselineBeatTracker
from src.models.advanced import RNNBeatTracker
from src.metrics.beat_tracking import BeatTrackingMetrics
from src.data.synthetic import SyntheticBeatDataset
from src.utils.device import get_device, set_seed
from src.utils.audio import extract_mel_spectrogram


class TestBaselineBeatTracker:
    """Test baseline beat tracker."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = BaselineBeatTracker()
        assert model.sample_rate == 22050
        assert model.tempo_min == 60
        assert model.tempo_max == 200
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = BaselineBeatTracker()
        
        # Create dummy audio
        duration = 5.0  # seconds
        sample_rate = 22050
        audio = np.random.randn(int(duration * sample_rate))
        
        # Forward pass
        result = model.forward(audio)
        
        assert "tempo" in result
        assert "beats" in result
        assert "onset_envelope" in result
        assert isinstance(result["tempo"], float)
        assert isinstance(result["beats"], np.ndarray)
    
    def test_predict(self):
        """Test prediction method."""
        model = BaselineBeatTracker()
        
        # Create dummy audio
        duration = 5.0
        sample_rate = 22050
        audio = np.random.randn(int(duration * sample_rate))
        
        # Predict
        tempo, beats = model.predict(audio)
        
        assert isinstance(tempo, float)
        assert isinstance(beats, np.ndarray)
        assert tempo > 0
        assert len(beats) > 0


class TestRNNBeatTracker:
    """Test RNN beat tracker."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RNNBeatTracker()
        assert model.input_dim == 128
        assert model.hidden_dim == 256
        assert model.num_layers == 2
        assert model.rnn_type == "LSTM"
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = RNNBeatTracker()
        
        # Create dummy mel spectrogram
        batch_size = 2
        seq_len = 100
        mel_bins = 128
        
        mel_spec = torch.randn(batch_size, seq_len, mel_bins)
        
        # Forward pass
        result = model.forward(mel_spec)
        
        assert "tempo_pred" in result
        assert "beat_pred" in result
        assert result["tempo_pred"].shape == (batch_size, seq_len)
        assert result["beat_pred"].shape == (batch_size, seq_len)
    
    def test_predict_beats(self):
        """Test beat prediction."""
        model = RNNBeatTracker()
        
        # Create dummy audio
        duration = 5.0
        sample_rate = 22050
        audio = np.random.randn(int(duration * sample_rate))
        
        # Predict beats
        result = model.predict_beats(audio)
        
        assert "tempo" in result
        assert "beats" in result
        assert isinstance(result["tempo"], float)
        assert isinstance(result["beats"], np.ndarray)


class TestBeatTrackingMetrics:
    """Test beat tracking metrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = BeatTrackingMetrics()
        assert metrics.tolerance_window == 0.07
        assert metrics.continuity_weight == 0.3
        assert metrics.accuracy_weight == 0.7
    
    def test_evaluate(self):
        """Test evaluation."""
        metrics = BeatTrackingMetrics()
        
        # Create dummy beats
        predicted_beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        ground_truth_beats = np.array([0.48, 0.98, 1.52, 1.98, 2.52])
        
        # Evaluate
        result = metrics.evaluate(predicted_beats, ground_truth_beats)
        
        assert "f_measure" in result
        assert "continuity" in result
        assert "accuracy" in result
        assert all(0 <= v <= 1 for v in result.values())
    
    def test_tempo_accuracy(self):
        """Test tempo accuracy calculation."""
        metrics = BeatTrackingMetrics()
        
        # Test accurate tempo
        accuracy = metrics.calculate_tempo_accuracy(120.0, 120.0)
        assert accuracy == 1.0
        
        # Test inaccurate tempo
        accuracy = metrics.calculate_tempo_accuracy(100.0, 120.0)
        assert accuracy == 0.0


class TestSyntheticDataset:
    """Test synthetic dataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = SyntheticBeatDataset(num_samples=10)
        assert len(dataset) == 10
        assert len(dataset.metadata) == 10
    
    def test_getitem(self):
        """Test dataset indexing."""
        dataset = SyntheticBeatDataset(num_samples=5)
        
        sample = dataset[0]
        
        assert "audio" in sample
        assert "beat_times" in sample
        assert "tempo" in sample
        assert "duration" in sample
        assert "sample_id" in sample
        
        assert isinstance(sample["audio"], np.ndarray)
        assert isinstance(sample["beat_times"], np.ndarray)
        assert isinstance(sample["tempo"], (int, float))
        assert isinstance(sample["duration"], (int, float))
        assert isinstance(sample["sample_id"], str)
    
    def test_metadata_generation(self):
        """Test metadata generation."""
        dataset = SyntheticBeatDataset(num_samples=5)
        
        metadata = dataset.metadata
        
        assert len(metadata) == 5
        assert "id" in metadata.columns
        assert "duration" in metadata.columns
        assert "tempo" in metadata.columns
        assert "beat_times" in metadata.columns
        
        # Check tempo range
        assert all(60 <= tempo <= 180 for tempo in metadata["tempo"])
        
        # Check duration range
        assert all(30 <= duration <= 180 for duration in metadata["duration"])


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy randomness
        np.random.seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2
    
    def test_extract_mel_spectrogram(self):
        """Test mel spectrogram extraction."""
        # Create dummy audio
        duration = 2.0
        sample_rate = 22050
        audio = np.random.randn(int(duration * sample_rate))
        
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(audio, sample_rate)
        
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == 128  # n_mels


if __name__ == "__main__":
    pytest.main([__file__])

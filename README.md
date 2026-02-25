# Beat Tracking Implementation

Research-focused implementation of beat tracking algorithms for music information retrieval, featuring both baseline and advanced neural network approaches.

## Overview

This project provides a comprehensive beat tracking system that can detect rhythmic beats in musical audio signals. It includes:

- **Baseline Implementation**: Librosa-based beat tracking using onset detection
- **Advanced Models**: RNN-based beat tracking with temporal modeling
- **Synthetic Dataset**: Generated musical audio with known beat patterns
- **Comprehensive Evaluation**: MIREX protocol metrics for beat tracking
- **Interactive Demo**: Streamlit-based web interface

## Features

### Models
- **BaselineBeatTracker**: Simple but effective librosa-based implementation
- **RNNBeatTracker**: Advanced RNN model with LSTM/GRU layers for temporal modeling

### Evaluation Metrics
- F-measure, Continuity, and Accuracy (MIREX protocol)
- Correct Metre Level (CML) and Allowed Metre Level (AML) metrics
- Tempo accuracy evaluation

### Data Pipeline
- Synthetic dataset generation with configurable parameters
- Support for different musical instruments and patterns
- Automatic train/validation/test splits

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Beat-Tracking-Implementation.git
cd Beat-Tracking-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Quick Start

### 1. Run the Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

This will open a web interface where you can:
- Upload audio files or use sample audio
- Select different models and parameters
- Visualize beat tracking results
- Download beat time annotations

### 2. Train Models

Train the baseline model:
```bash
python scripts/train.py model=baseline
```

Train the RNN model:
```bash
python scripts/train.py model=rnn
```

### 3. Evaluate Models

Run comprehensive evaluation:
```bash
python scripts/evaluate.py
```

This will generate a leaderboard comparing different models.

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/baseline.yaml`: Baseline model parameters
- `configs/model/rnn.yaml`: RNN model parameters
- `configs/data/synthetic.yaml`: Dataset generation parameters

### Example Configuration Override

```bash
python scripts/train.py model=rnn training.epochs=50 data.generation.num_samples=2000
```

## Project Structure

```
beat-tracking-implementation/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Dataset classes
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training utilities
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── data/                  # Data directory
├── outputs/               # Training outputs
├── checkpoints/           # Model checkpoints
└── assets/                # Generated assets
```

## Usage Examples

### Basic Beat Tracking

```python
from src.models.baseline import BaselineBeatTracker
from src.utils.audio import load_audio

# Load audio
audio, sr = load_audio("path/to/audio.wav")

# Create model
model = BaselineBeatTracker(sample_rate=sr)

# Predict beats
tempo, beats = model.predict(audio)

print(f"Estimated tempo: {tempo:.1f} BPM")
print(f"Detected {len(beats)} beats")
```

### Advanced RNN Model

```python
from src.models.advanced import RNNBeatTracker
import torch

# Create model
model = RNNBeatTracker(
    input_dim=128,
    hidden_dim=256,
    num_layers=2,
    rnn_type="LSTM"
)

# Load checkpoint
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Predict beats
result = model.predict_beats(audio)
tempo = result["tempo"]
beats = result["beats"]
```

### Evaluation Metrics

```python
from src.metrics.beat_tracking import BeatTrackingMetrics

# Create metrics calculator
metrics = BeatTrackingMetrics(tolerance_window=0.07)

# Evaluate predictions
results = metrics.evaluate(predicted_beats, ground_truth_beats)

print(f"F-measure: {results['f_measure']:.3f}")
print(f"Continuity: {results['continuity']:.3f}")
print(f"Accuracy: {results['accuracy']:.3f}")
```

## Dataset

The project includes a synthetic dataset generator that creates musical audio with known beat patterns. This is useful for:

- Training and evaluation without requiring real audio data
- Controlled experiments with known ground truth
- Testing algorithm robustness

### Dataset Features

- Configurable tempo ranges (60-180 BPM)
- Multiple instrument types (kick, snare, hi-hat, bass, melody)
- Variable audio durations (30-180 seconds)
- Optional reverb and noise effects
- Automatic train/validation/test splits

## Evaluation

The evaluation follows the MIREX (Music Information Retrieval Evaluation eXchange) protocol:

### Metrics

- **F-measure**: Overall beat tracking performance
- **Continuity**: Longest continuous segment accuracy
- **Accuracy**: Timing accuracy within tolerance window
- **CMLC/CMLT**: Correct Metre Level Continuity/Tracking
- **AMLC/AMLT**: Allowed Metre Level Continuity/Tracking

### Leaderboard

The evaluation generates a comprehensive leaderboard comparing different models:

| Model | Samples | Tempo Accuracy | F-Measure | Continuity | Accuracy |
|-------|---------|----------------|-----------|------------|----------|
| baseline | 100 | 0.850 | 0.723 ± 0.156 | 0.689 ± 0.178 | 0.712 ± 0.142 |
| rnn | 100 | 0.892 | 0.756 ± 0.134 | 0.734 ± 0.152 | 0.748 ± 0.128 |

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{beat_tracking_implementation,
  title={Beat Tracking Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Beat-Tracking-Implementation}
}
```

## Acknowledgments

- Librosa for audio processing utilities
- PyTorch for deep learning framework
- MIREX for evaluation protocol
- Streamlit for interactive demo interface

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**⚠️ IMPORTANT: Please read the [Privacy & Ethics Disclaimer](DISCLAIMER.md) before using this software.**
# Beat-Tracking-Implementation

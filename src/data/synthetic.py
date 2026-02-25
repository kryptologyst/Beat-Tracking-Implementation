"""Synthetic dataset generation for beat tracking evaluation."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticBeatDataset(Dataset):
    """Synthetic dataset for beat tracking evaluation.
    
    This dataset generates synthetic musical audio with known beat patterns
    for training and evaluating beat tracking algorithms.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        duration_range: Tuple[float, float] = (30.0, 180.0),
        tempo_range: Tuple[int, int] = (60, 180),
        time_signature: Tuple[int, int] = (4, 4),
        sample_rate: int = 22050,
        instrument_types: List[str] = None,
        volume_range: Tuple[float, float] = (0.3, 1.0),
        reverb_prob: float = 0.3,
        noise_prob: float = 0.2,
        noise_level: float = 0.1,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
    ):
        """Initialize synthetic beat dataset.
        
        Args:
            num_samples: Number of samples to generate.
            duration_range: Range of audio durations in seconds.
            tempo_range: Range of tempos in BPM.
            time_signature: Time signature (numerator, denominator).
            sample_rate: Sample rate of generated audio.
            instrument_types: Types of instruments to use.
            volume_range: Range of volume levels.
            reverb_prob: Probability of adding reverb.
            noise_prob: Probability of adding noise.
            noise_level: Level of added noise.
            data_dir: Directory to save generated data.
            split: Dataset split ('train', 'val', 'test').
        """
        self.num_samples = num_samples
        self.duration_range = duration_range
        self.tempo_range = tempo_range
        self.time_signature = time_signature
        self.sample_rate = sample_rate
        self.instrument_types = instrument_types or ["kick", "snare", "hihat", "bass", "melody"]
        self.volume_range = volume_range
        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.split = split
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wav_dir = self.data_dir / "wav"
        self.wav_dir.mkdir(exist_ok=True)
        
        # Generate metadata
        self.metadata = self._generate_metadata()
        
        logger.info(f"Initialized SyntheticBeatDataset with {num_samples} samples")
    
    def _generate_metadata(self) -> pd.DataFrame:
        """Generate metadata for the dataset.
        
        Returns:
            DataFrame containing sample metadata.
        """
        metadata = []
        
        for i in range(self.num_samples):
            # Random parameters
            duration = random.uniform(*self.duration_range)
            tempo = random.randint(*self.tempo_range)
            volume = random.uniform(*self.volume_range)
            
            # Calculate beat times
            beat_interval = 60.0 / tempo  # seconds per beat
            num_beats = int(duration / beat_interval)
            beat_times = np.arange(num_beats) * beat_interval
            
            # Add some randomness to beat times
            beat_times += np.random.normal(0, 0.02, len(beat_times))  # 20ms jitter
            beat_times = np.clip(beat_times, 0, duration)
            
            metadata.append({
                "id": f"synthetic_{self.split}_{i:04d}",
                "duration": duration,
                "tempo": tempo,
                "volume": volume,
                "beat_times": beat_times.tolist(),
                "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
                "sample_rate": self.sample_rate,
            })
        
        return pd.DataFrame(metadata)
    
    def _generate_audio(self, metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic audio for a single sample.
        
        Args:
            metadata: Sample metadata.
            
        Returns:
            Tuple of (audio_array, beat_times).
        """
        duration = metadata["duration"]
        tempo = metadata["tempo"]
        volume = metadata["volume"]
        beat_times = np.array(metadata["beat_times"])
        
        # Generate audio
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples)
        
        # Generate different instrument tracks
        for instrument in self.instrument_types:
            track = self._generate_instrument_track(
                duration=duration,
                tempo=tempo,
                beat_times=beat_times,
                instrument=instrument,
                volume=volume,
            )
            audio += track
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) * volume
        
        # Add reverb
        if random.random() < self.reverb_prob:
            audio = self._add_reverb(audio)
        
        # Add noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_level, len(audio))
            audio += noise
        
        # Final normalization
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio, beat_times
    
    def _generate_instrument_track(
        self,
        duration: float,
        tempo: float,
        beat_times: np.ndarray,
        instrument: str,
        volume: float,
    ) -> np.ndarray:
        """Generate a track for a specific instrument.
        
        Args:
            duration: Duration in seconds.
            tempo: Tempo in BPM.
            beat_times: Beat time locations.
            instrument: Instrument type.
            volume: Volume level.
            
        Returns:
            Generated audio track.
        """
        num_samples = int(duration * self.sample_rate)
        track = np.zeros(num_samples)
        
        if instrument == "kick":
            # Kick drum on every beat
            for beat_time in beat_times:
                sample_idx = int(beat_time * self.sample_rate)
                if sample_idx < num_samples:
                    kick = self._generate_kick_drum()
                    end_idx = min(sample_idx + len(kick), num_samples)
                    track[sample_idx:end_idx] += kick[:end_idx-sample_idx]
        
        elif instrument == "snare":
            # Snare on beats 2 and 4
            for i, beat_time in enumerate(beat_times):
                if i % 4 in [1, 3]:  # Beats 2 and 4
                    sample_idx = int(beat_time * self.sample_rate)
                    if sample_idx < num_samples:
                        snare = self._generate_snare_drum()
                        end_idx = min(sample_idx + len(snare), num_samples)
                        track[sample_idx:end_idx] += snare[:end_idx-sample_idx]
        
        elif instrument == "hihat":
            # Hi-hat on every beat
            for beat_time in beat_times:
                sample_idx = int(beat_time * self.sample_rate)
                if sample_idx < num_samples:
                    hihat = self._generate_hihat()
                    end_idx = min(sample_idx + len(hihat), num_samples)
                    track[sample_idx:end_idx] += hihat[:end_idx-sample_idx]
        
        elif instrument == "bass":
            # Bass line following chord progression
            track = self._generate_bass_line(duration, tempo, beat_times)
        
        elif instrument == "melody":
            # Melodic line
            track = self._generate_melody(duration, tempo, beat_times)
        
        return track * volume
    
    def _generate_kick_drum(self) -> np.ndarray:
        """Generate kick drum sound."""
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
        
        # Low frequency sine wave with envelope
        freq = 60  # Hz
        t = np.linspace(0, duration, samples)
        envelope = np.exp(-t * 20)  # Exponential decay
        
        kick = np.sin(2 * np.pi * freq * t) * envelope
        return kick
    
    def _generate_snare_drum(self) -> np.ndarray:
        """Generate snare drum sound."""
        duration = 0.05  # 50ms
        samples = int(duration * self.sample_rate)
        
        # Noise with envelope
        t = np.linspace(0, duration, samples)
        envelope = np.exp(-t * 30)
        
        snare = np.random.normal(0, 0.5, samples) * envelope
        return snare
    
    def _generate_hihat(self) -> np.ndarray:
        """Generate hi-hat sound."""
        duration = 0.02  # 20ms
        samples = int(duration * self.sample_rate)
        
        # High frequency noise
        hihat = np.random.normal(0, 0.3, samples)
        return hihat
    
    def _generate_bass_line(self, duration: float, tempo: float, beat_times: np.ndarray) -> np.ndarray:
        """Generate bass line."""
        num_samples = int(duration * self.sample_rate)
        track = np.zeros(num_samples)
        
        # Simple bass pattern
        bass_freq = 80  # Hz
        for i, beat_time in enumerate(beat_times):
            if i % 2 == 0:  # Every other beat
                sample_idx = int(beat_time * self.sample_rate)
                note_duration = 60.0 / tempo / 2  # Half note
                note_samples = int(note_duration * self.sample_rate)
                
                if sample_idx < num_samples:
                    end_idx = min(sample_idx + note_samples, num_samples)
                    t = np.linspace(0, note_duration, end_idx - sample_idx)
                    note = np.sin(2 * np.pi * bass_freq * t)
                    track[sample_idx:end_idx] += note
        
        return track
    
    def _generate_melody(self, duration: float, tempo: float, beat_times: np.ndarray) -> np.ndarray:
        """Generate melodic line."""
        num_samples = int(duration * self.sample_rate)
        track = np.zeros(num_samples)
        
        # Simple melody pattern
        melody_freqs = [220, 247, 262, 294, 330, 349, 392, 440]  # A3 to A4
        
        for i, beat_time in enumerate(beat_times):
            sample_idx = int(beat_time * self.sample_rate)
            note_duration = 60.0 / tempo / 4  # Quarter note
            note_samples = int(note_duration * self.sample_rate)
            
            if sample_idx < num_samples:
                end_idx = min(sample_idx + note_samples, num_samples)
                freq = melody_freqs[i % len(melody_freqs)]
                t = np.linspace(0, note_duration, end_idx - sample_idx)
                note = np.sin(2 * np.pi * freq * t) * 0.3
                track[sample_idx:end_idx] += note
        
        return track
    
    def _add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Add simple reverb effect."""
        # Simple delay-based reverb
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        reverb = np.zeros_like(audio)
        
        for i in range(len(audio)):
            if i >= delay_samples:
                reverb[i] = audio[i] + 0.3 * audio[i - delay_samples]
            else:
                reverb[i] = audio[i]
        
        return reverb
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, float, List]]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing audio, beats, and metadata.
        """
        metadata = self.metadata.iloc[idx]
        
        # Generate audio
        audio, beat_times = self._generate_audio(metadata)
        
        return {
            "audio": audio,
            "beat_times": beat_times,
            "tempo": metadata["tempo"],
            "duration": metadata["duration"],
            "sample_id": metadata["id"],
        }
    
    def save_sample(self, idx: int, output_dir: Optional[Path] = None) -> None:
        """Save a sample to disk.
        
        Args:
            idx: Sample index.
            output_dir: Output directory.
        """
        if output_dir is None:
            output_dir = self.wav_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample = self[idx]
        metadata = self.metadata.iloc[idx]
        
        # Save audio
        audio_path = output_dir / f"{metadata['id']}.wav"
        sf.write(audio_path, sample["audio"], self.sample_rate)
        
        # Save beat annotations
        beat_path = output_dir / f"{metadata['id']}_beats.txt"
        np.savetxt(beat_path, sample["beat_times"], fmt="%.6f")
        
        logger.info(f"Saved sample {metadata['id']} to {output_dir}")
    
    def save_all_samples(self, output_dir: Optional[Path] = None) -> None:
        """Save all samples to disk.
        
        Args:
            output_dir: Output directory.
        """
        for i in range(len(self)):
            self.save_sample(i, output_dir)
        
        # Save metadata
        if output_dir is None:
            output_dir = self.data_dir
        
        metadata_path = output_dir / f"metadata_{self.split}.csv"
        self.metadata.to_csv(metadata_path, index=False)
        
        logger.info(f"Saved all {len(self)} samples to {output_dir}")

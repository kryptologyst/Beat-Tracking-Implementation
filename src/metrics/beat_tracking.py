"""Evaluation metrics for beat tracking following MIREX protocol."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class BeatTrackingMetrics:
    """Beat tracking evaluation metrics following MIREX protocol.
    
    This class implements the standard evaluation metrics for beat tracking
    as used in the Music Information Retrieval Evaluation eXchange (MIREX).
    """
    
    def __init__(
        self,
        tolerance_window: float = 0.07,  # 70ms tolerance
        continuity_weight: float = 0.3,
        accuracy_weight: float = 0.7,
    ):
        """Initialize beat tracking metrics.
        
        Args:
            tolerance_window: Tolerance window in seconds for beat matching.
            continuity_weight: Weight for continuity metric.
            accuracy_weight: Weight for accuracy metric.
        """
        self.tolerance_window = tolerance_window
        self.continuity_weight = continuity_weight
        self.accuracy_weight = accuracy_weight
        
        logger.info(f"Initialized BeatTrackingMetrics with tolerance={tolerance_window}s")
    
    def evaluate(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
        audio_duration: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate beat tracking performance.
        
        Args:
            predicted_beats: Predicted beat times in seconds.
            ground_truth_beats: Ground truth beat times in seconds.
            audio_duration: Duration of audio in seconds.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return {
                "f_measure": 0.0,
                "continuity": 0.0,
                "accuracy": 0.0,
                "cmlc": 0.0,
                "cmlt": 0.0,
                "amlc": 0.0,
                "amlt": 0.0,
            }
        
        # Calculate individual metrics
        f_measure = self._calculate_f_measure(predicted_beats, ground_truth_beats)
        continuity = self._calculate_continuity(predicted_beats, ground_truth_beats)
        accuracy = self._calculate_accuracy(predicted_beats, ground_truth_beats)
        
        # Calculate CMLC and CMLT (Correct Metre Level Continuity/Tracking)
        cmlc, cmlt = self._calculate_cml_metrics(predicted_beats, ground_truth_beats)
        
        # Calculate AMLC and AMLT (Allowed Metre Level Continuity/Tracking)
        amlc, amlt = self._calculate_aml_metrics(predicted_beats, ground_truth_beats)
        
        return {
            "f_measure": f_measure,
            "continuity": continuity,
            "accuracy": accuracy,
            "cmlc": cmlc,
            "cmlt": cmlt,
            "amlc": amlc,
            "amlt": amlt,
        }
    
    def _calculate_f_measure(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> float:
        """Calculate F-measure for beat tracking.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            F-measure value.
        """
        # Find matches within tolerance window
        matches = self._find_matches(predicted_beats, ground_truth_beats)
        
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return 0.0
        
        precision = len(matches) / len(predicted_beats)
        recall = len(matches) / len(ground_truth_beats)
        
        if precision + recall == 0:
            return 0.0
        
        f_measure = 2 * precision * recall / (precision + recall)
        return f_measure
    
    def _calculate_continuity(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> float:
        """Calculate continuity metric.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            Continuity value.
        """
        if len(predicted_beats) == 0:
            return 0.0
        
        # Find matches
        matches = self._find_matches(predicted_beats, ground_truth_beats)
        
        if len(matches) == 0:
            return 0.0
        
        # Calculate longest continuous segment
        match_indices = [i for i, (p, g) in enumerate(matches)]
        longest_segment = self._find_longest_continuous_segment(match_indices)
        
        continuity = longest_segment / len(ground_truth_beats)
        return continuity
    
    def _calculate_accuracy(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> float:
        """Calculate accuracy metric.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            Accuracy value.
        """
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return 0.0
        
        # Find matches
        matches = self._find_matches(predicted_beats, ground_truth_beats)
        
        if len(matches) == 0:
            return 0.0
        
        # Calculate timing accuracy
        timing_errors = []
        for pred_beat, gt_beat in matches:
            error = abs(pred_beat - gt_beat)
            timing_errors.append(error)
        
        # Accuracy based on timing errors
        mean_error = np.mean(timing_errors)
        accuracy = max(0, 1 - mean_error / self.tolerance_window)
        
        return accuracy
    
    def _calculate_cml_metrics(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate Correct Metre Level metrics.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            Tuple of (CMLC, CMLT) values.
        """
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return 0.0, 0.0
        
        # Estimate tempo from ground truth
        gt_tempo = self._estimate_tempo(ground_truth_beats)
        
        # Generate correct metre level beats
        cml_beats = self._generate_metre_level_beats(ground_truth_beats, gt_tempo, level=1)
        
        # Evaluate against CML beats
        cmlc = self._calculate_continuity(predicted_beats, cml_beats)
        cmlt = self._calculate_accuracy(predicted_beats, cml_beats)
        
        return cmlc, cmlt
    
    def _calculate_aml_metrics(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate Allowed Metre Level metrics.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            Tuple of (AMLC, AMLT) values.
        """
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return 0.0, 0.0
        
        # Estimate tempo from ground truth
        gt_tempo = self._estimate_tempo(ground_truth_beats)
        
        # Generate allowed metre level beats (2x and 0.5x tempo)
        aml_beats = []
        for multiplier in [0.5, 1.0, 2.0]:
            level_beats = self._generate_metre_level_beats(
                ground_truth_beats, gt_tempo, level=multiplier
            )
            aml_beats.extend(level_beats)
        
        aml_beats = np.array(sorted(aml_beats))
        
        # Evaluate against AML beats
        amlc = self._calculate_continuity(predicted_beats, aml_beats)
        amlt = self._calculate_accuracy(predicted_beats, aml_beats)
        
        return amlc, amlt
    
    def _find_matches(
        self,
        predicted_beats: np.ndarray,
        ground_truth_beats: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Find matches between predicted and ground truth beats.
        
        Args:
            predicted_beats: Predicted beat times.
            ground_truth_beats: Ground truth beat times.
            
        Returns:
            List of (predicted_beat, ground_truth_beat) pairs.
        """
        matches = []
        used_gt_indices = set()
        
        for pred_beat in predicted_beats:
            # Find closest ground truth beat within tolerance
            distances = np.abs(ground_truth_beats - pred_beat)
            closest_idx = np.argmin(distances)
            
            if (distances[closest_idx] <= self.tolerance_window and
                closest_idx not in used_gt_indices):
                matches.append((pred_beat, ground_truth_beats[closest_idx]))
                used_gt_indices.add(closest_idx)
        
        return matches
    
    def _find_longest_continuous_segment(self, indices: List[int]) -> int:
        """Find the longest continuous segment in a list of indices.
        
        Args:
            indices: List of indices.
            
        Returns:
            Length of longest continuous segment.
        """
        if not indices:
            return 0
        
        indices = sorted(indices)
        longest = 1
        current = 1
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
        
        return max(longest, current)
    
    def _estimate_tempo(self, beats: np.ndarray) -> float:
        """Estimate tempo from beat times.
        
        Args:
            beats: Beat times in seconds.
            
        Returns:
            Estimated tempo in BPM.
        """
        if len(beats) < 2:
            return 120.0  # Default tempo
        
        # Calculate inter-beat intervals
        intervals = np.diff(beats)
        
        # Remove outliers
        intervals = intervals[intervals > 0.1]  # Remove very short intervals
        
        if len(intervals) == 0:
            return 120.0
        
        # Estimate tempo from median interval
        median_interval = np.median(intervals)
        tempo = 60.0 / median_interval
        
        return tempo
    
    def _generate_metre_level_beats(
        self,
        original_beats: np.ndarray,
        tempo: float,
        level: float,
    ) -> np.ndarray:
        """Generate beats at a specific metre level.
        
        Args:
            original_beats: Original beat times.
            tempo: Tempo in BPM.
            level: Metre level (0.5, 1.0, 2.0, etc.).
            
        Returns:
            Generated beat times.
        """
        if len(original_beats) == 0:
            return np.array([])
        
        # Calculate beat interval at the specified level
        beat_interval = 60.0 / (tempo * level)
        
        # Generate beats
        start_time = original_beats[0]
        end_time = original_beats[-1]
        
        beats = np.arange(start_time, end_time + beat_interval, beat_interval)
        
        return beats
    
    def calculate_tempo_accuracy(
        self,
        predicted_tempo: float,
        ground_truth_tempo: float,
        tolerance: float = 0.05,  # 5% tolerance
    ) -> float:
        """Calculate tempo accuracy.
        
        Args:
            predicted_tempo: Predicted tempo in BPM.
            ground_truth_tempo: Ground truth tempo in BPM.
            tolerance: Relative tolerance for tempo matching.
            
        Returns:
            Tempo accuracy (1.0 if within tolerance, 0.0 otherwise).
        """
        relative_error = abs(predicted_tempo - ground_truth_tempo) / ground_truth_tempo
        
        if relative_error <= tolerance:
            return 1.0
        else:
            return 0.0

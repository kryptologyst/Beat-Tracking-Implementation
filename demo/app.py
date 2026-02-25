"""Streamlit demo for beat tracking."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from ..models.baseline import BaselineBeatTracker
from ..models.advanced import RNNBeatTracker
from ..utils.audio import extract_mel_spectrogram, load_audio
from ..utils.device import get_device, get_device_info
from ..utils.logging import setup_logging

# Set up logging
setup_logging(level="WARNING")  # Reduce logging noise in demo
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Beat Tracking Implementation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ Privacy & Ethics Disclaimer</h4>
    <p><strong>This is a research demonstration only.</strong> This beat tracking implementation is designed for educational and research purposes. It does not collect, store, or transmit any personal information. Audio files are processed locally and not shared with any external services.</p>
    <p><strong>Prohibited Uses:</strong> This tool must not be used for biometric identification, voice cloning, or any form of personal surveillance. Misuse of audio processing technology for unauthorized purposes is strictly prohibited.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎵 Beat Tracking Implementation</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Baseline (Librosa)", "RNN-based"],
    help="Choose between baseline librosa implementation or advanced RNN model"
)

# Model parameters
st.sidebar.subheader("Model Parameters")

sample_rate = st.sidebar.slider(
    "Sample Rate",
    min_value=8000,
    max_value=48000,
    value=22050,
    step=1000,
    help="Sample rate for audio processing"
)

hop_length = st.sidebar.slider(
    "Hop Length",
    min_value=256,
    max_value=1024,
    value=512,
    step=128,
    help="Hop length for analysis"
)

tempo_range = st.sidebar.slider(
    "Tempo Range (BPM)",
    min_value=30,
    max_value=300,
    value=(60, 200),
    help="Expected tempo range"
)

# Advanced parameters
if model_type == "RNN-based":
    st.sidebar.subheader("RNN Parameters")
    
    hidden_dim = st.sidebar.slider(
        "Hidden Dimension",
        min_value=64,
        max_value=512,
        value=256,
        step=64,
    )
    
    num_layers = st.sidebar.slider(
        "Number of Layers",
        min_value=1,
        max_value=4,
        value=2,
    )
    
    dropout = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.1,
    )

# Initialize models
@st.cache_resource
def load_models():
    """Load and cache models."""
    device = get_device("auto")
    
    models = {}
    
    # Baseline model
    models["baseline"] = BaselineBeatTracker(
        sample_rate=sample_rate,
        hop_length=hop_length,
        tempo_min=tempo_range[0],
        tempo_max=tempo_range[1],
    )
    
    # RNN model
    if model_type == "RNN-based":
        models["rnn"] = RNNBeatTracker(
            input_dim=128,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            sample_rate=sample_rate,
            hop_length=hop_length,
            tempo_min=tempo_range[0],
            tempo_max=tempo_range[1],
        )
    
    return models, device

# Load models
models, device = load_models()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Input Audio")
    
    # Audio input options
    input_method = st.radio(
        "Choose Input Method",
        ["Upload File", "Record Audio", "Use Sample"],
        horizontal=True,
    )
    
    audio_data = None
    audio_file = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload an audio file for beat tracking analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_file = tmp_file.name
            
            try:
                audio_data, sr = load_audio(audio_file, sample_rate=sample_rate)
                st.success(f"✅ Loaded audio: {uploaded_file.name}")
                st.info(f"Duration: {len(audio_data) / sr:.2f} seconds")
            except Exception as e:
                st.error(f"❌ Error loading audio: {e}")
    
    elif input_method == "Record Audio":
        st.info("🎤 Audio recording functionality would be implemented here")
        st.warning("This feature requires additional setup for microphone access")
    
    elif input_method == "Use Sample":
        # Generate a sample audio
        st.info("🎵 Generating sample audio...")
        
        # Create a simple beat pattern
        duration = 10  # seconds
        tempo = 120  # BPM
        beat_interval = 60.0 / tempo
        
        # Generate audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.zeros_like(t)
        
        # Add kick drum on every beat
        for i in range(int(duration / beat_interval)):
            beat_time = i * beat_interval
            beat_samples = int(beat_time * sample_rate)
            if beat_samples < len(audio_data):
                # Simple kick drum sound
                kick_duration = 0.1
                kick_samples = int(kick_duration * sample_rate)
                kick_end = min(beat_samples + kick_samples, len(audio_data))
                kick_sound = np.sin(2 * np.pi * 60 * t[beat_samples:kick_end]) * np.exp(-t[beat_samples:kick_end] * 20)
                audio_data[beat_samples:kick_end] += kick_sound
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        st.success("✅ Generated sample audio with 120 BPM")

with col2:
    st.header("🎯 Beat Tracking Results")
    
    if audio_data is not None:
        # Process audio
        if st.button("🔍 Analyze Beats", type="primary"):
            with st.spinner("Analyzing beats..."):
                try:
                    # Select model
                    if model_type == "Baseline (Librosa)":
                        model = models["baseline"]
                        result = model.predict(audio_data)
                        tempo, beats = result
                        
                        # Get additional info
                        full_result = model.forward(audio_data)
                        onset_env = full_result["onset_envelope"]
                        
                    else:  # RNN-based
                        model = models["rnn"]
                        result = model.predict_beats(audio_data)
                        tempo = result["tempo"]
                        beats = result["beats"]
                        onset_env = result.get("beat_pred", np.zeros(100))
                    
                    # Display results
                    st.success("✅ Beat tracking completed!")
                    
                    # Metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        st.metric("Estimated Tempo", f"{tempo:.1f} BPM")
                    
                    with col_metrics2:
                        st.metric("Number of Beats", len(beats))
                    
                    with col_metrics3:
                        if len(beats) > 0:
                            beat_interval = np.mean(np.diff(beats))
                            st.metric("Beat Interval", f"{beat_interval:.3f}s")
                        else:
                            st.metric("Beat Interval", "N/A")
                    
                    # Visualization
                    st.subheader("📊 Visualization")
                    
                    # Create time axis
                    duration = len(audio_data) / sample_rate
                    time_axis = np.linspace(0, duration, len(audio_data))
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=["Audio Waveform", "Onset Envelope", "Detected Beats"],
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.3, 0.3],
                    )
                    
                    # Audio waveform
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=audio_data,
                            mode="lines",
                            name="Audio",
                            line=dict(color="blue", width=1),
                        ),
                        row=1, col=1,
                    )
                    
                    # Onset envelope
                    onset_time = np.linspace(0, duration, len(onset_env))
                    fig.add_trace(
                        go.Scatter(
                            x=onset_time,
                            y=onset_env,
                            mode="lines",
                            name="Onset Envelope",
                            line=dict(color="green", width=2),
                        ),
                        row=2, col=1,
                    )
                    
                    # Detected beats
                    if len(beats) > 0:
                        beat_y = np.ones(len(beats)) * np.max(onset_env)
                        fig.add_trace(
                            go.Scatter(
                                x=beats,
                                y=beat_y,
                                mode="markers",
                                name="Detected Beats",
                                marker=dict(color="red", size=8, symbol="triangle-up"),
                            ),
                            row=3, col=1,
                        )
                    
                    # Update layout
                    fig.update_layout(
                        height=600,
                        showlegend=False,
                        title="Beat Tracking Analysis",
                    )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
                    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
                    fig.update_yaxes(title_text="Onset Strength", row=2, col=1)
                    fig.update_yaxes(title_text="Beat Detection", row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Beat times table
                    if len(beats) > 0:
                        st.subheader("📍 Beat Times")
                        
                        # Create DataFrame
                        import pandas as pd
                        beat_df = pd.DataFrame({
                            "Beat Number": range(1, len(beats) + 1),
                            "Time (seconds)": beats,
                            "Time (mm:ss)": [f"{int(t//60):02d}:{t%60:05.2f}" for t in beats],
                        })
                        
                        st.dataframe(beat_df, use_container_width=True)
                        
                        # Download results
                        csv = beat_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Beat Times (CSV)",
                            data=csv,
                            file_name="beat_times.csv",
                            mime="text/csv",
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    logger.error(f"Analysis error: {e}")
    
    else:
        st.info("👆 Please provide audio input to analyze beats")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Beat Tracking Implementation - Research Demo</p>
    <p>Built with Streamlit, PyTorch, and Librosa</p>
</div>
""", unsafe_allow_html=True)

# Device info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("System Info")

device_info = get_device_info()
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write(f"**CUDA:** {'✅' if device_info['cuda'] else '❌'}")
st.sidebar.write(f"**MPS:** {'✅' if device_info['mps'] else '❌'}")

if device_info['cuda']:
    st.sidebar.write(f"**GPU:** {device_info['cuda_device_name']}")
    st.sidebar.write(f"**GPU Count:** {device_info['cuda_device_count']}")

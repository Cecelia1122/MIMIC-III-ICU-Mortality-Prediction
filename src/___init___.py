"""
MIMIC-III ICU Mortality/Length-of-Stay Prediction Package

A multimodal deep learning framework for ICU outcome prediction using: 
- Vital signs (time-series)
- Laboratory tests
- Demographics
- Diagnoses

Supports: 
- In-hospital mortality prediction
- Length-of-stay prediction
- 48-hour mortality prediction

Models:
- LSTM (for temporal data)
- Transformer (for temporal data)
- MLP (baseline)
- Multimodal fusion
"""

__version__ = '1.0.0'
__author__ = 'Xue Li'

from .data_extraction import MIMICDataExtractor
from .preprocessing import MIMICPreprocessor
from . dataset import MIMICDataset, get_data_loaders
from .models import (
    LSTMModel,
    TransformerModel,
    MLPModel,
    MultimodalFusionModel,
    get_model
)
from .train import Trainer, compute_class_weights
from .evaluate import Evaluator
from .utils import set_seed, get_device

__all__ = [
    'MIMICDataExtractor',
    'MIMICPreprocessor',
    'MIMICDataset',
    'get_data_loaders',
    'LSTMModel',
    'TransformerModel',
    'MLPModel',
    'MultimodalFusionModel',
    'get_model',
    'Trainer',
    'compute_class_weights',
    'Evaluator',
    'set_seed',
    'get_device',
]
"""
Knowledge-driven Quality Assessment of Labeled Sensor Data

A research implementation for identifying label errors in EEG data using
weak supervision and domain-specific labeling functions.

Main Components:
- LabelingFunctions: 20+ domain-specific EEG pattern classification functions
- TimeSeriesFeaturizer: Advanced time-series feature extraction
- dataset: Data loading and expert annotation preprocessing utilities
"""

__version__ = "1.0.0"
__author__ = "Mononito Goswami, Benedikt Boecking, Patrick J. Coppler, Jonathan Elmer, Artur Dubrawski"

# Import main classes for easy access
from .labeling_functions import LabelingFunctions
from .featurizer import TimeSeriesFeaturizer
from . import dataset
from . import utils

__all__ = [
    "LabelingFunctions",
    "TimeSeriesFeaturizer",
    "dataset",
    "utils"
]
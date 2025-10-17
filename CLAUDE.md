# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a knowledge-driven quality assessment system for labeled EEG (Electroencephalography) sensor data. The project uses weak supervision and domain-specific labeling functions to identify and assess label errors in neurological sensor data.

## Core Architecture

### Data Processing Pipeline
1. **Data Loading** (`identify_label_errors/dataset.py`): Loads EEG time-series data and expert annotations from CSV/Excel files
2. **Feature Extraction** (`identify_label_errors/featurizer.py`): Extracts complex time-series features using signal processing techniques
3. **Labeling Functions** (`identify_label_errors/labeling_functions.py`): Applies 20+ domain-specific labeling functions based on neurological expertise
4. **Weak Supervision** (`identify_label_errors/weak_supervision.py`): Combines multiple labeling sources using probabilistic methods
5. **Evaluation** (`identify_label_errors/evaluation.py`): Assesses label quality and identifies potential errors

### Key Components

**LabelingFunctions Class** (`identify_label_errors/labeling_functions.py`):
- Implements knowledge-driven labeling based on EEG signal characteristics
- Uses Gaussian Mixture Models, peak detection, and baseline analysis
- Provides 20 specialized labeling functions for different EEG patterns
- Categories: Normal, Suppressed, Suppressed with Ictal, Burst Suppression

**TimeSeriesFeaturizer Class** (`identify_label_errors/featurizer.py`):
- Extracts features from raw EEG signals using advanced signal processing
- Implements winsorization, robust line fitting, peak detection
- Handles missing data and artifacts in sensor readings

**Configuration System** (`config.yaml`):
- Centralizes all parameters for data paths, labeling thresholds, and processing settings
- Expert label descriptions map clinical terminology to computational categories
- Labeling function parameters control sensitivity of detection algorithms

## Development Commands

### Running the Analysis
```bash
# Load and explore data interactively
jupyter notebook explore_data.ipynb

# Run Python modules directly
python -c "from identify_label_errors import dataset; data, labels = dataset.load_data()"
```

### Configuration
- Modify `config.yaml` to adjust:
  - Data file paths (`data_path`, `expert_labels_path`)
  - Labeling function parameters (thresholds, tolerances)
  - Expert label mappings and descriptions
  - Verbosity and explanation settings

### Key Parameters in config.yaml
- `near_zero`: Threshold for near-zero amplitude detection (1 mV)
- `eeg_high_*`: High amplitude thresholds (15, 10, 5, 4 mV)
- `eeg_low_2`: Low amplitude baseline threshold (2 mV)
- `n_high_amplitude_peaks_per_hour`: Spike detection sensitivity (12)
- `splits`: Number of analysis windows for temporal segmentation (6)

## Code Structure

### Package Organization
```
identify_label_errors/
├── dataset.py          # Data loading and preprocessing
├── labeling_functions.py # 20+ domain-specific labeling functions
├── featurizer.py       # Time-series feature extraction
├── weak_supervision.py # Probabilistic label combination
├── evaluation.py       # Label quality assessment
└── utils.py           # Configuration and utility functions
```

### Signal Processing Workflow
1. **Preprocessing**: Winsorization, artifact removal, NaN handling
2. **Baseline Analysis**: Robust line fitting to extract trends
3. **Peak Detection**: Identify amplitude spikes with prominence thresholds
4. **Distribution Modeling**: Gaussian Mixture Models for bimodal patterns
5. **Classification**: Rule-based labeling using neurological domain knowledge

### Label Categories
- **Normal**: High baseline amplitude (>4-10 mV), stable patterns
- **Suppressed**: Low amplitude (<2 mV), minimal activity
- **Suppressed with Ictal**: Low baseline with high-amplitude spikes (>7.5 mV)
- **Burst Suppression**: Alternating suppression and burst patterns
- **Abstain**: Insufficient confidence for classification

## Data Dependencies
- EEG time-series data in CSV format with temporal measurements
- Expert annotations in Excel format with clinical classifications
- Proper column naming and timestamp alignment between datasets
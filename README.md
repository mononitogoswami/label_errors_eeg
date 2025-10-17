<div align="center">

# Knowledge-driven Quality Assessment of Labeled Sensor Data

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)

</div>

## 🔥 News
- **[2025]** Code and experiments for reproducing "Knowledge-driven Quality Assessment of Labeled Sensor Data" are now available!

## 📖 Introduction

Machine Learning (ML) models have been successfully applied to intelligently process multivariate sensor data. However, most popular ML models rely on a generous supply of expert-annotated reference data for training. Annotating ample amounts of raw sensor data is not only tedious, but also expensive and prone to error. To economize, researchers often consider using legacy, already labeled databases to develop their algorithms. Some application domains, however, evolve over time and the old labels may become irrelevant or inconsistent with current practice. This issue manifests itself very clearly in healthcare, where standards of clinical practice, diagnosis, and treatment change frequently, and where consequences of using the models trained on outdated annotations of otherwise relevant reference data may be grave. We propose an affordable knowledge-driven approach to auditing the quality and currency of annotated sensor data and demonstrate its utility on an important clinical problem using electroencephalographic waveform data. Our novel contributions include: (1) defining labeling functions directly on time series summaries rather than requiring manual pointillistic annotation, and (2) using model-driven adjudication to systematically identify annotation errors. Using only 17 expert-defined heuristics developed in a few hours, our label model achieved moderate agreement with original expert labels (Cohen's $\kappa=0.559$) across 7,000+ hours of EEG data, with per-class sensitivities of 0.859 for continuous background, 0.760 for generalized suppression, and 0.665 for burst suppression with epileptiform activity. Systematic adjudication of 31 patients revealed that 100\% were misclassified in the original annotations due to evolving clinical nomenclature, demonstrating that our approach can effectively detect systematic annotation errors stemming from outdated reference standards.

**Key Contributions:**
1. **Domain Knowledge Encoding**: 20+ specialized labeling functions that capture clinical EEG interpretation expertise
2. **Weak Supervision Framework**: Probabilistic approach to combine multiple noisy labeling sources
3. **Advanced Signal Processing**: Robust feature extraction pipeline for EEG time-series analysis

### 🧠 EEG Pattern Classification: One Framework, Multiple Clinical States

<div align="center">

| **Clinical State** | **Characteristics** | **Clinical Significance** |
|:---:|:---:|:---:|
| **Normal** | Stable high-amplitude baseline (>4-10 mV) | Healthy brain activity |
| **Suppressed** | Low-amplitude activity (<2 mV) | Reduced consciousness |
| **Suppressed with Ictal** | Low baseline + high spikes (>7.5 mV) | Seizure activity |
| **Burst Suppression** | Alternating suppression/burst patterns | Deep sedation/coma |

</div>

Our approach automatically identifies inconsistencies between expert labels and signal characteristics across these critical neurological states.

### 🏗️ Architecture in a Nutshell

Our framework processes EEG signals through a multi-stage pipeline that combines signal processing with domain expertise:

1. **Signal Preprocessing**: Artifact removal, amplitude normalization, and missing data handling
2. **Feature Extraction**: 28 comprehensive time-series features including baseline trends, spectral properties, and complexity measures
3. **Domain Knowledge Encoding**: 20+ labeling functions that capture clinical EEG interpretation rules
4. **Weak Supervision**: Probabilistic combination of labeling function outputs to identify potential label errors

### 📊 Labeling Functions Capture Clinical Expertise

Our approach encodes neurological domain knowledge through specialized functions:

**Baseline Analysis:**
- `high_aeeg_baseline_normal()`: Detects stable high-amplitude baselines indicating normal activity
- `low_baseline_aeeg()`: Identifies suppressed patterns with consistently low amplitudes
- `bimodal_aeeg()`: Analyzes bimodal amplitude distributions characteristic of burst-suppression

**Spike Detection:**
- `very_spiky_aeeg_suppressed_with_ictal()`: High-amplitude spike detection for seizure identification
- `well_separated_aeeg_modes_suppressed_with_ictal()`: Modal separation analysis for ictal patterns

**Pattern Recognition:**
- `aeeg_NOT_near_zero_normal()`: Normal activity detection based on amplitude thresholds
- `high_baseline_frequent_drops()`: Burst-suppression pattern recognition through amplitude variations

## 🧑‍💻 Usage

**Recommended Python Version:** Python 3.8+ (tested on Python 3.8, 3.9, 3.10, 3.11).

You can install this package directly from GitHub:
```bash
git clone https://github.com/autonlab/label_errors_eeg.git
cd label_errors_eeg
pip install -r requirements.txt
```

### Basic Usage

**EEG Label Quality Assessment**
```python
from identify_label_errors import dataset, LabelingFunctions

# Load EEG data and expert annotations
data, expert_labels = dataset.load_data()
print(f"Loaded {data.shape[0]} EEG samples with {expert_labels.shape[0]} expert annotations")

# Apply labeling functions to assess label quality
eeg_signal = data.iloc[0].values  # First EEG sample
lf = LabelingFunctions(eeg_signal)
vote_vector = lf.get_vote_vector()
lf_names = lf.get_LF_names()

# Display labeling function results
for name, vote in zip(lf_names, vote_vector):
    if vote != 'Abstain':
        print(f"{name}: {vote}")
```

**Advanced Feature Extraction**
```python
from identify_label_errors import TimeSeriesFeaturizer

# Extract comprehensive time-series features
featurizer = TimeSeriesFeaturizer(eeg_signal, remove_artifacts=True)
features = featurizer.featurize()
feature_names = featurizer.get_feature_names()

print(f"Extracted {len(features)} features:")
for name, value in zip(feature_names[:5], features[:5]):  # Show first 5
    print(f"  {name}: {value:.3f}")
```

**Signal Processing Utilities**
```python
from identify_label_errors.utils import sample_entropy, higuchi_fd

# Complexity measures for signal analysis
entropy = sample_entropy(eeg_signal, order=2)
fractal_dim = higuchi_fd(eeg_signal, kmax=6)
print(f"Sample entropy: {entropy:.3f}, Higuchi FD: {fractal_dim:.3f}")
```

## ⚙️ Configuration

The framework uses `config.yaml` for centralized parameter management, allowing easy customization for different EEG systems and clinical settings:

```yaml
# Data Configuration
data_path: 'datasets/output_all_wide_502.csv'
expert_labels_path: 'datasets/eeg_PUH_mater_eeg_file_deidentified.xlsx'

# Clinical Thresholds (customizable per institution)
labeling_function_parameters:
  near_zero: 1                    # Near-zero amplitude threshold (mV)
  eeg_high_15: 15                # Very high amplitude threshold (mV)
  eeg_high_10: 10                # High amplitude threshold (mV)
  eeg_low_2: 2                   # Low amplitude baseline (mV)
  n_high_amplitude_peaks_per_hour: 12  # Spike detection sensitivity

# Expert Knowledge Encoding
expert_description:
  S1: 'High shaggy aEEG baseline (constantly at 4-200 mV)'
  S2: 'Low aEEG baseline continually at low amplitude <= 2-3 mV.'
  S3: 'aEEG baseline never falls to near-zero (< 1mV).'
  S4: 'Abrupt, recurring and high amplitude (> 7.5 mV) spikes.'
  S5: 'spiky aEEG with higher baseline but frequent falls to near-zero'
```

**Customization Options:**
- **Clinical Thresholds**: Adjust amplitude thresholds for different EEG acquisition systems
- **Detection Sensitivity**: Modify spike detection parameters for specific patient populations
- **Data Sources**: Update file paths for different datasets or institutions
- **Verbosity Control**: Enable detailed explanations for clinical review

## Labeling Functions

The system implements 20+ specialized labeling functions based on neurological expertise:

### Baseline Analysis Functions
- `high_aeeg_baseline_normal()`: Detects stable high-amplitude baselines
- `low_baseline_aeeg()`: Identifies suppressed baseline patterns
- `bimodal_aeeg()`: Analyzes bimodal amplitude distributions

### Spike Detection Functions
- `very_spiky_aeeg_suppressed_with_ictal()`: High-amplitude spike detection
- `well_separated_aeeg_modes_suppressed_with_ictal()`: Modal separation analysis

### Pattern Recognition Functions
- `aeeg_NOT_near_zero_normal()`: Normal activity detection
- `high_baseline_frequent_drops()`: Burst-suppression pattern recognition

Each function applies specific clinical criteria and returns one of the standard labels or abstains if criteria aren't met.

## Advanced Features

### Feature Extraction

The `TimeSeriesFeaturizer` class extracts 28 comprehensive features:

```python
from identify_label_errors.featurizer import TimeSeriesFeaturizer

featurizer = TimeSeriesFeaturizer(eeg_signal, remove_artifacts=True)
features = featurizer.featurize()
feature_names = featurizer.get_feature_names()

print(f"Extracted features: {len(features)}")
for name, value in zip(feature_names, features):
    print(f"  {name}: {value:.3f}")
```

### Signal Processing Utilities

```python
from identify_label_errors.utils import sample_entropy, higuchi_fd

# Complexity measures
entropy = sample_entropy(eeg_signal, order=2)
fractal_dim = higuchi_fd(eeg_signal, kmax=6)
print(f"Sample entropy: {entropy:.3f}, Higuchi FD: {fractal_dim:.3f}")
```

## Data Requirements

### Input Data Format

**EEG Time-series** (`datasets/output_all_wide_502.csv`):
- CSV format with temporal EEG measurements
- Each row represents one time window (typically 1-hour segments)
- Columns contain amplitude values at regular time intervals
- Missing values handled automatically with interpolation

**Expert Annotations** (`datasets/eeg_PUH_mater_eeg_file_deidentified.xlsx`):
- Excel format with clinical expert labels
- Required columns: `id`, `Timestamp`, `Background`, `Superimposed patterns`, `Reactivity`
- Background values: 1=suppressed, 2=burst-suppression, 3-4=continuous
- Superimposed patterns: 0=none, 1=seizure, 2=myoclonic, 6-7=epileptiform

### Data Preprocessing

The system automatically handles:
- Missing value imputation using forward fill
- Timestamp alignment between EEG and annotations
- Artifact removal using winsorization (1% trimming)
- Amplitude normalization and outlier detection

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -e .

# Code formatting
black identify_label_errors/
flake8 identify_label_errors/
```

### Adding New Labeling Functions

1. Implement function in `LabelingFunctions` class following naming convention
2. Add function to `get_vote_vector()` method
3. Include descriptive name in `get_LF_names()` method
4. Update configuration parameters if needed
5. Add tests and documentation

## 🪪 License

MIT License

Copyright (c) 2025 Auton Lab, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/label_errors_eeg/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png">
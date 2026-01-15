# MIMIC-III-ICU-Mortality-Prediction
# MIMIC-III ICU Mortality Prediction

A multimodal deep learning framework for predicting in-hospital mortality of ICU patients using clinical time-series data and static features from the MIMIC-III database. 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PhysioNet](https://img.shields.io/badge/PhysioNet-Credentialed-green.svg)

## Overview

This project implements an end-to-end pipeline for ICU mortality prediction using the MIMIC-III (Medical Information Mart for Intensive Care III) database. It combines static patient features (demographics, aggregated vitals/labs) with temporal clinical data (time-series of vital signs and laboratory tests) using a multimodal fusion architecture.

### Key Features

- **Multimodal architecture**: Fuses static features with temporal sequences
- **Multiple model support**: MLP, LSTM, Transformer, and Multimodal Fusion
- **Comprehensive preprocessing**: Automated extraction from raw MIMIC-III CSV files
- **Clinical relevance**: Uses first 48 hours of ICU data for early prediction
- **Class imbalance handling**: Weighted loss functions for imbalanced mortality data

### Clinical Task

**Binary classification**: Predict whether an ICU patient will survive or die during their hospital stay, using only the first 48 hours of ICU data.

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Survived | Patient discharged alive |
| 1 | Deceased | In-hospital mortality |

## Results

### Model Comparison

| Model | Input Data | AUC-ROC | Recall (Deceased) | F1 Score |
|-------|------------|---------|-------------------|----------|
| **Multimodal** | Static + Time-series | **0.8941** | **86.26%** | Best overall |
| MLP | Static only | 0.8893 | 79.81% | Strong baseline |
| Transformer | Time-series only | 0.8551 | 73.63% | Captures temporal patterns |
| LSTM | Time-series only | 0.8077 | 80.22% | Trend-sensitive |

### Key Findings

- **Multimodal fusion achieves best performance** (AUC 0.894) by combining static and temporal information
- **Static features provide strong baseline** — demographics and aggregated statistics are highly predictive
- **Transformer captures critical fluctuations** in vital signs through attention mechanisms
- **High recall (86%)** is crucial for clinical applications — minimizing missed mortality cases

### ROC Curve

<p align="center">
  <img src="results/figures/roc_curve.png" alt="ROC Curve" width="500"/>
</p>

## Dataset

### MIMIC-III Database

This project uses the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/), which contains:
- **46,520** ICU stays
- **38,597** distinct patients
- **58,976** hospital admissions

**Access Requirements**:  MIMIC-III requires credentialed access through PhysioNet.  You must complete the CITI training program and sign a data use agreement. 

### Cohort Selection Criteria

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| Age | ≥ 18 years | Adult patients only |
| ICU LOS | ≥ 24 hours | Sufficient data for prediction |
| ICU LOS | ≤ 30 days | Exclude long-term/outlier stays |
| ICU admission | First only | Avoid data leakage |

**Final cohort**: ~33,000 ICU stays after filtering

### Features Extracted

#### Static Features (56 dimensions)

| Category | Features |
|----------|----------|
| Demographics | Age, gender, ethnicity (one-hot), admission type |
| Vital signs (aggregated) | Heart rate, SBP, DBP, MAP, respiratory rate, temperature, SpO2, GCS — each with mean, std, min, max, first, last |
| Laboratory tests (aggregated) | Glucose, potassium, sodium, chloride, bicarbonate, BUN, creatinine, hemoglobin, hematocrit, WBC, platelets, lactate — each with mean, std, min, max, first, last |

#### Time-Series Features (12 timesteps × 20 features)

- **Temporal resolution**: 4-hour intervals over 48 hours = 12 timesteps
- **Vital signs**: 8 features (heart rate, BP, respiratory rate, temperature, SpO2, GCS)
- **Laboratory tests**: 12 features (glucose, electrolytes, renal function, CBC, lactate)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Access to MIMIC-III database (PhysioNet credentialed)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mimic-icu-mortality.git
cd mimic-icu-mortality

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### MIMIC-III Data Setup

1. Obtain credentialed access at [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Download and extract MIMIC-III CSV files
3. Place files in `./data/mimic-iii-clinical-database-1.4/`

Expected directory structure:
```
data/
└── mimic-iii-clinical-database-1.4/
    ├── PATIENTS.csv. gz
    ├── ADMISSIONS.csv.gz
    ├── ICUSTAYS.csv.gz
    ├── CHARTEVENTS.csv.gz
    ├── LABEVENTS.csv.gz
    └── DIAGNOSES_ICD. csv.gz
```

## Usage

### Quick Start

```bash
# Full pipeline:  preprocess → train → evaluate
python main.py --mode full --mimic_path ./data/mimic-iii-clinical-database-1.4
```

### Step-by-Step

#### 1. Data Extraction & Preprocessing

```bash
python main.py --mode preprocess \
    --mimic_path ./data/mimic-iii-clinical-database-1.4 \
    --hours 48 \
    --interval 4
```

This will:
- Extract cohort based on inclusion criteria
- Extract vital signs and lab tests for first 48 hours
- Create aggregated static features
- Create resampled time-series (4-hour intervals)
- Split into train/val/test sets (70/10/20)
- Save processed data to `./data/processed/`

#### 2. Train Model

```bash
# Train multimodal model (recommended)
python main.py --mode train --model multimodal --epochs 50

# Train other models
python main.py --mode train --model mlp --epochs 50
python main.py --mode train --model lstm --epochs 50
python main. py --mode train --model transformer --epochs 50
```

#### 3. Evaluate Model

```bash
python main.py --mode evaluate --model multimodal
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `preprocess`, `train`, `evaluate`, or `full` |
| `--model` | `multimodal` | `mlp`, `lstm`, `transformer`, or `multimodal` |
| `--mimic_path` | `./data/mimic-iii-clinical-database-1.4` | Path to MIMIC-III data |
| `--data_dir` | `./data/processed` | Path to processed data |
| `--hours` | `48` | Hours of data to extract |
| `--interval` | `4` | Time-series resampling interval (hours) |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--save_dir` | `./checkpoints` | Checkpoint directory |
| `--results_dir` | `./results` | Results directory |

## Project Structure

```
mimic-icu-mortality/
├── main.py                     # Unified entry point
├── requirements.txt            # Dependencies
├── README.md
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_extraction.py      # MIMIC-III data extraction
│   ├── preprocessing.py        # Feature engineering
│   ├── dataset.py              # PyTorch Dataset classes
│   ├── models.py               # Model architectures
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics
│   └── utils.py                # Utility functions
├── checkpoints/                # Saved models
├── results/
│   └── figures/                # Plots and visualizations
└── data/
    ├── mimic-iii-clinical-database-1.4/  # Raw MIMIC-III (not included)
    └── processed/              # Processed datasets
        ├── train. pkl
        ├── val.pkl
        ├── test.pkl
        ├── feature_info.json
        └── scaler.pkl
```

## Model Architectures

### 1. MLP (Baseline)

Simple feedforward network for static features only. 

```
Input (56) → Linear(256) → BN → ReLU → Dropout
          → Linear(128) → BN → ReLU → Dropout
          → Linear(64)  → BN → ReLU → Dropout
          → Linear(2)   → Output
```

### 2. LSTM

Bidirectional LSTM for time-series data.

```
Input (12, 20) → BiLSTM(128, 2 layers) → Last hidden state
              → Linear(128) → ReLU → Dropout
              → Linear(2)   → Output
```

### 3. Transformer

Transformer encoder for time-series with positional encoding.

```
Input (12, 20) → Linear projection (128)
              → Positional Encoding
              → TransformerEncoder (4 heads, 2 layers)
              → Global Average Pooling
              → Linear(64) → ReLU → Dropout
              → Linear(2)  → Output
```

### 4. Multimodal Fusion (Best)

Combines static and temporal branches with late fusion.

```
Static Branch: 
  Input (56) → Linear(128) → BN → ReLU → Dropout → Linear(64) → Features (64)

Temporal Branch:
  Input (12, 20) → BiLSTM(128, 2 layers) → Features (256)

Fusion:
  Concat(64 + 256) → Linear(128) → BN → ReLU → Dropout
                   → Linear(64)  → ReLU → Dropout
                   → Linear(2)   → Output
```

## API Usage

```python
from src import (
    MIMICDataExtractor,
    MIMICPreprocessor,
    get_data_loaders,
    get_model,
    Trainer,
    Evaluator
)

# Extract data
extractor = MIMICDataExtractor('./data/mimic-iii-clinical-database-1.4')
cohort, vitals, labs, diagnoses = extractor.extract_all(hours=48)

# Preprocess
preprocessor = MIMICPreprocessor()
preprocessor.prepare_dataset(
    cohort, vitals, labs, diagnoses,
    output_dir='./data/processed'
)

# Load data
train_loader, val_loader, test_loader = get_data_loaders(
    './data/processed',
    batch_size=64,
    use_static=True,
    use_time_series=True
)

# Create model
model = get_model(
    model_type='multimodal',
    static_dim=56,
    ts_input_dim=20,
    num_classes=2,
    device='cuda'
)

# Train
trainer = Trainer(model, train_loader, val_loader, device='cuda')
history = trainer.train(num_epochs=50)

# Evaluate
evaluator = Evaluator(model, test_loader, device='cuda')
metrics = evaluator.full_evaluation('./results/figures')
```

## Technical Notes

### Why Multimodal Fusion Works Best

1. **Complementary information**: Static features capture patient baseline, while time-series captures clinical trajectory
2. **Different time scales**:  Aggregated statistics summarize 48 hours; time-series shows temporal patterns
3. **Redundancy reduction**: Late fusion allows each branch to learn specialized representations

### Handling Class Imbalance

ICU mortality is inherently imbalanced (~13% mortality rate in MIMIC-III). We address this with: 
- **Weighted cross-entropy loss**: Inverse frequency weighting
- **Stratified splits**: Maintain class distribution across train/val/test
- **Evaluation metrics**: Focus on AUC-ROC and Recall, not just accuracy

### Missing Data Handling

Medical time-series data is inherently sparse.  Our approach:
- **Static features**: Median imputation
- **Time-series**: Forward fill → Backward fill → Zero fill
- **Normalization**: Z-score normalization (fit on training data only)

## Comparison with Published Benchmarks

| Method | AUC-ROC | Reference |
|--------|---------|-----------|
| **Our Multimodal** | **0.894** | This work |
| Harutyunyan et al. (2019) | 0.870 | Multitask benchmark |
| SAPS-II Score | 0.780 | Clinical baseline |
| APACHE-II Score | 0.750 | Clinical baseline |

Our multimodal approach achieves competitive results with state-of-the-art methods. 

## Future Improvements

- [ ] Attention-based multimodal fusion
- [ ] Incorporate clinical notes (NLP)
- [ ] Multi-task learning (mortality + LOS + readmission)
- [ ] Interpretability analysis (feature importance, attention visualization)
- [ ] External validation on eICU/MIMIC-IV

## References

- Johnson, A.  E., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*. 
- Harutyunyan, H., et al. (2019). Multitask learning and benchmarking with clinical time series data. *Scientific Data*.
- Purushotham, S., et al. (2018). Benchmarking deep learning models on large healthcare datasets. *Journal of Biomedical Informatics*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Xue Li**  
MSc Communications and Signal Processing, Imperial College London  
Email: xueli.xl1122@gmail.com  
PhysioNet Credentialed Researcher ✓

## Acknowledgments

- PhysioNet and the MIMIC-III team for providing open-access clinical data
- PyTorch team for the deep learning framework
- MIT Laboratory for Computational Physiology

---

**Note**: This project requires credentialed access to MIMIC-III. The raw data is not included in this repository.  Please follow PhysioNet's data use agreement. 

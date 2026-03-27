# Fraud Detection Using Behavioral and Transactional Features

MSc Dissertation — Computing and Data Science (CDS)

## Overview

This project investigates whether combining bot-detection behavioural signals with traditional transactional features improves fraud detection in digital payment systems. A synthetic dataset calibrated from the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile-money simulator is used to train and evaluate multiple classification models under realistic class-imbalance conditions.

## Repository Structure

```
thesis/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── Thesis.md                        # Dissertation document (Markdown)
│
├── data/
│   └── PS_20174392719_…_log.csv     # PaySim dataset
│
├── behavioral_dataset.py            # Synthetic behavioral feature generation
├── fraud_dataset.py                 # Combined synthetic dataset generation (calibrated from PaySim)
├── paysim_dataset.py                # PaySim parameter extraction (distributions, fraud rates)
├── generate_dataset_tables.py       # Dataset samples and statistics for thesis
│
├── training_pipeline.py             # Reusable ML pipeline: split, encode, train, threshold-tune, evaluate
├── run_all_models.py                # Orchestrator: multi-seed, multi-algorithm experiments
├── generate_test_graphs.py          # Test-set performance graphs and literature comparison
├── validate_distributions.py        # Statistical validation: KS tests, QQ plots, distribution comparison
│
├── behavioral_model_training.py     # Standalone: behavioral-only model
├── transactional_model_training.py  # Standalone: transactional-only model
├── combined_model_training.py       # Standalone: combined model
│
└── results/                         # Generated outputs (CSV tables, PNG plots, Markdown reports)
    ├── all_seeds_summary.csv
    ├── aggregated_results.csv
    ├── test_set_summary.csv / .md
    ├── literature_comparison.csv / .md
    ├── sample_*.csv                   # Dataset samples for thesis
    ├── stats_*.csv                    # Descriptive statistics
    ├── pr_curves_comparison.png
    ├── pr_curves_by_feature_set.png
    ├── roc_curves_comparison.png
    ├── performance_bar_chart.png
    ├── decision_distribution.png
    ├── score_distribution_*.png
    ├── confusion_matrix_*.png
    ├── feature_importance_*.png
    ├── threshold_sensitivity_*.png
    └── dist_*.png                     # Distribution validation plots
```

## Dataset

The PaySim dataset must be downloaded separately:

1. Download `PS_20174392719_1491204439457_log.csv` from [Kaggle](https://www.kaggle.com/code/kartik2112/fraud-detection-on-paysim-dataset/data?select=PS_20174392719_1491204439457_log.csv)
2. Place it in `thesis/data/`

The synthetic dataset is generated programmatically by `fraud_dataset.py` using statistical parameters extracted from PaySim via `paysim_dataset.py`.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- scikit-learn 1.8+
- pandas, numpy, matplotlib, tabulate
- Faker (for synthetic data generation)

## Reproducing Results

### Complete Execution Order

To fully replicate all results in the thesis, execute scripts in this order:

#### 1. Generate Dataset Tables for Thesis

```bash
python generate_dataset_tables.py
```

Creates sample datasets and descriptive statistics used in Section 5.1.1 of the thesis. Outputs to `results/`:

- `sample_*.csv` — Sample behavioral, transactional, and combined datasets
- `stats_*.csv` — Descriptive statistics for each feature set
- Markdown tables for direct inclusion in thesis

#### 2. Validate Synthetic Data Distributions

```bash
python validate_distributions.py
```

Compares synthetic data distributions against PaySim using KS tests, QQ plots, and fraud rate analysis. Generates both unimodal and multimodal comparisons to demonstrate the improvement. Outputs plots and statistics to `results/`:

- `dist_*_unimodal.png` — Unimodal approach plots
- `dist_*_multimodal.png` — Multimodal approach plots (current method)
- KS test comparison showing improvement

#### 3. Run Full Experiment Suite (Multi-Seed)

```bash
python run_all_models.py
```

**Primary experiment** - trains 4 algorithms × 3 feature sets × 5 random seeds = 60 model configurations under precision ≥ 0.50 constraint. Outputs:

- `results/all_seeds_summary.csv` — per-seed metrics (used for Tables 5.5, 5.6, 5.7)
- `results/aggregated_results.csv` — mean ± std across seeds (used for Table 5.4)
- `results/test_set_summary.csv` — test set metrics for seed=42
- Precision-constraint sweep analysis (Section 5.3.2)

**Expected runtime:** ~1–2 hours depending on hardware.

#### 4. Generate Test Set Performance Graphs

```bash
python generate_test_graphs.py
```

Runs 12 configurations (seed=42) and produces all visualization plots with precision ≥ 0.50 constraint:

- `threshold_sensitivity_*.png` — Threshold sensitivity curves with 0.50 constraint line
- `pr_curves_*.png` — Precision-recall curves
- `roc_curves_*.png` — ROC curves
- `confusion_matrix_*.png` — Confusion matrices
- `feature_importance_*.png` — Feature importance rankings
- `decision_distribution.png` — Three-tier decision distribution with T_low, T_op, and T_high annotations
- `score_distribution_*.png` — Score distribution histograms with threshold lines
- `literature_comparison.md` — Updated with Combined RF and GB results (Table 5.13)
- `test_set_summary.md` — Test set summary table

#### 5. Run Individual Models (Optional)

```bash
python behavioral_model_training.py
python transactional_model_training.py
python combined_model_training.py
```

For debugging or testing individual feature sets. All use precision ≥ 0.50 constraint.

### Critical Notes for Replication

1. **Precision Constraint**: All experiments use precision ≥ 0.50 (not 0.90) based on the constraint sweep analysis (Section 5.3.2)
2. **Random Seeds**: Multi-seed experiments use seeds [42, 123, 456, 789, 2024] for statistical robustness
3. **Data Dependencies**: PaySim dataset must be in `data/` directory before running any scripts
4. **Output Order**: Scripts generate outputs incrementally - later scripts depend on earlier outputs

## Algorithms

| Algorithm               | Role                               |
| ----------------------- | ---------------------------------- |
| DummyClassifier (prior) | Baseline — predicts majority class |
| Logistic Regression     | Linear baseline                    |
| Random Forest           | Primary ensemble method            |
| Gradient Boosting       | Secondary ensemble method          |

## Three-Tier Decision System

The pipeline produces a fraud probability score per transaction and maps it to an operational decision using three thresholds tuned on the validation set:

| Score Range            | Decision   | Action                                |
| ---------------------- | ---------- | ------------------------------------- |
| Score < T_low          | **ALLOW**  | Auto-approve (low risk, recall ≥ 95%) |
| T_low ≤ Score < T_high | **REVIEW** | Manual review queue (recall < 95%)    |
| Score ≥ T_high         | **DENY**   | Auto-block (precision ≥ 70%)          |

- **T_high (DENY threshold, precision ≥ 0.70)**: The lowest threshold where at least 70% of flagged transactions are truly fraudulent. Transactions scoring ≥ T_high are automatically denied.
- **T_op (Operational threshold, precision ≥ 0.50)**: The highest-recall threshold that still maintains at least 50% precision. This represents the recommended operating point that balances detection coverage with false positive control.
- **T_low (REVIEW threshold, recall ≥ 0.95)**: The highest threshold that captures at least 95% of known fraud. Transactions scoring between T_low and T_high are flagged for manual review, ensuring that most fraudulent cases are captured within the review or denial tiers.

## Key Findings

### Main Results (Precision ≥ 0.50 Constraint)

- **Combined models significantly improve fraud detection**: Adding behavioral features to transactional features increases PR-AUC by ~38% for both RF (0.583 vs 0.419) and GB (0.584 vs 0.426)
- **Recall improvements dramatic**: Combined GB achieves 0.524 ± 0.065 vs Transactional GB's 0.134 ± 0.104 — nearly 4× improvement
- **Behavioral features excel at bot detection**: Behavioral RF achieves ROC-AUC 0.919 ± 0.015 and PR-AUC 0.852 ± 0.026
- **Stealth bot detection**: Under precision ≥ 0.50, stealth bot recall reaches 0.947 ± 0.042 (RF), exceeding overall bot recall

### Three-Tier Threshold System

- **T_high**: Precision ≥ 0.70 for automatic denial
- **T_op**: Precision ≥ 0.50 for operational reference point
- **T_low**: Recall ≥ 0.95 for manual review queue

## Hardware

Experiments were run on:

- MacBook Pro (16-inch, 2021)
- Apple M1 Max
- 32 GB RAM

## License

This project is part of an MSc dissertation and is not licensed for commercial use.

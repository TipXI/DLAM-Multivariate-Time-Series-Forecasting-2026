# Deep Learning for Multivariate Operations Time Series Forecasting (SS26)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/HuggingFace-Leaderboard-yellow)](https://aiml-tuda-dlam-ts-project-leaderboard-2026.hf.space/)

**Course:** Deep Learning (SS26), Technical University of Darmstadt  
**Instructors:** Prof. Kristian Kersting, Maurice Kraus, Ruben Härle  
**Group:** Grex XXIII (Group 23)  
**Members:** Adel Bukseisa, Mariam Farhat, Mehshid Atiq, Omar Elmorsy  

---

## Overview

This repository contains the complete, reproducible implementation of our deep learning forecasting system for the **DLAM Multivariate Time Series Benchmark 2026**. 

The goal is to forecast the future **336-hour (14-day)** operational load index across **96 heterogeneous operations units**. Our framework integrates:
* **PyTorch Deep Operations ResNet (`DeepOperationsNet`)**: Incorporates learned entity embeddings, LayerNorm, GELU, and residual MLP connections trained directly with an aligned $\ell_1$ loss.
* **Feature Engineering & Missingness Robustness**: Multi-scale forward rolling cumulative pressure representations ($3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$), binary missingness masks, and domain-specific physical interaction ratios.
* **Consensus Ensembling**: Blends deep neural representations with multi-configuration gradient-boosted decision trees.
* **Uncertainty Quantification**: Split Conformal Prediction providing distribution-free prediction intervals with exact statistical coverage guarantees ($80\%, 90\%, 95\%$).
* **Domain Generalization**: Cross-domain transfer on real **Binance Crypto Hourly Trading Volume** (Julien, 2023 via `kagglehub`), achieving a **>40% error reduction** (`3.12%` vs. `5.23%` WAPE) over statistical baselines.

---

## Official Public Leaderboard Benchmark & CSV Mapping

The table below reports the **exact scores from the official Hugging Face public validation leaderboard** (scored automatically against hidden validation labels) along with the corresponding generated CSV files in `submissions/`:

| Model / Baseline | Source | Generated CSV File | MAE $\downarrow$ | MSE $\downarrow$ | RMSE $\downarrow$ | MAPE (\%) $\downarrow$ | sMAPE (\%) $\downarrow$ | WAPE (\%) $\downarrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `naive_last_value` | Official Lecturer Baseline | `submissions/baselines/naive_last_value.csv` | `5.2945` | `48.6110` | `6.9722` | `61.35` | `52.71` | **`48.10`** |
| `lag24_repeat` | Official Lecturer Baseline | `submissions/baselines/lag24_repeat.csv` | `5.1539` | `43.5946` | `6.6026` | `78.51` | `47.31` | **`46.82`** |
| `lag168_repeat` | Official Lecturer Baseline | `submissions/baselines/lag168_repeat.csv` | `5.1405` | `45.1725` | `6.7210` | `71.87` | `47.28` | **`46.70`** |
| `seasonal_mean` | Official Lecturer Baseline | `submissions/baselines/seasonal_mean.csv` | `3.7876` | `27.1029` | `5.2060` | `45.68` | `37.43` | **`34.41`** |
| `tide_v1` (TiDE) | Group 23 Model | `submissions/tide.csv` | `3.6939` | `26.8609` | `5.1828` | `40.34` | `37.33` | **`33.56`** |
| `ensemble_v1` | Group 23 Model | `submissions/ensemble_v1.csv` | `1.6823` | `9.5873` | `3.0963` | `19.82` | `16.97` | **`15.28`** |
| `ensemble_v2` (`nuhhh_v2`) | Group 23 Model | `submissions/ensemble_v2.csv` | `1.5779` | **`9.3050`** | **`3.0504`** | `18.51` | `15.88` | **`14.33`** |
| `grand_master_v3` | Group 23 Model | `submissions/grand_master_v3.csv` | `1.5792` | `9.3261` | `3.0539` | `18.41` | `15.84` | **`14.35`** |
| **Winning Model (`nuhhh_v5`)** | **Group 23 Model** | `submissions/predictions.csv` | **`1.5739`** | `9.3137` | `3.0518` | **`18.35`** | **`15.81`** | **`14.30`** |

> **Key Takeaway:** Our winning model (`nuhhh_v5`) achieves a **$>3.3\times$ error reduction** over the official naive baseline (`48.10%`) and a **$>2.4\times$ error reduction** over the official seasonal mean baseline (`34.41%`), setting our top score of **14.30% WAPE** (14.298% WAPE, MAE `1.5739`).

---

## Repository Structure

```text
├── Group_23.pdf                      # Project Exposé defining methodology and roadmap
├── requirements.txt                  # Full project dependencies for reproducibility
├── README.md                         # Project documentation and reproduction guide
├── final_submission.zip              # Container-ready model archive for private evaluation
│
├── src/                              # Core library modules
│   ├── deep_net.py                   # PyTorch DeepOperationsNet with entity embeddings
│   ├── tide.py                       # TiDE architecture implementation
│   ├── dlinear.py                    # DLinear structural baseline
│   ├── revin.py                      # Reversible Instance Normalization layer
│   ├── features.py                   # Feature pipeline & missingness masking
│   ├── conformal.py                  # Split Conformal Prediction module
│   ├── dataset.py                    # Vectorized sliding-window PyTorch dataset
│   ├── trainer.py                    # Training loop with L1 loss & Cosine Annealing
│   └── evaluation.py                 # Official competition metrics (WAPE, MAE, etc.)
│
├── scripts/                          # Automated execution scripts
│   ├── download_data.py              # Acquires dataset files from Hugging Face
│   ├── run_and_evaluate_baselines.py # Evaluates all 4 course baselines
│   ├── train_top_ensemble.py         # Trains the winning top-tier ensemble
│   ├── run_ablations.py              # Automated ablation study suite
│   ├── run_conformal_prediction.py   # Calibrates distribution-free uncertainty intervals
│   └── run_crypto_experiment.py      # Additional dataset generalization experiment
│
├── results/                          # Structured empirical results
│   ├── ablations/                    # Architecture, loss function, and feature ablations
│   ├── conformal/                    # Conformal coverage calibration metrics
│   └── additional_dataset/           # Crypto generalization benchmark results
│
├── submissions/                      # Generated submission prediction CSVs
│   ├── baselines/                    # naive_last_value, seasonal_mean, lag24, lag168
│   ├── predictions.csv               # Top winning leaderboard submission nuhhh_v5 (14.30% WAPE)
│   ├── ensemble_v2.csv               # Winning leaderboard submission nuhhh_v2 (14.33% WAPE)
│   ├── ensemble_v1.csv               # Initial ensemble submission (15.28% WAPE)
│   ├── tide.csv                      # TiDE standalone submission (33.56% WAPE)
│   └── dlinear.csv                   # DLinear standalone submission
│
├── student/                          # Official template directory
│   ├── report_template/              # Complete 4-6 page academic LaTeX paper & bib
│   └── submission_template/          # Verified offline inference script & checkpoint.pt
```

---

## Installation & Setup

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/TipXI/DLAM-Multivariate-Time-Series-Forecasting-2026.git
cd DLAM-Multivariate-Time-Series-Forecasting-2026

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Fast Inference Quickstart (Skip Training in < 3s)

If you wish to generate predictions immediately without training models from scratch, you can run inference directly using our pre-trained checkpoint to reproduce our top winning leaderboard submission (`nuhhh_v5`, **14.30% WAPE** / 14.298%):

```bash
# 1. Download benchmark input data (~5s)
python scripts/download_data.py

# 2. Run instant inference from pre-trained winning ensemble checkpoint (~2.8s)
python student/submission_template/predict.py \
  --input_dir data \
  --output_file submissions/predictions.csv \
  --checkpoint student/submission_template/checkpoint.pt
```
*This produces `submissions/predictions.csv` with all 32,256 required rows, 0 nulls, exactly matching our top winning 14.30% WAPE leaderboard submission.*

---

## Full Reproduction Guide (Training from Scratch)

### Step 1: Download Benchmark Data
Acquires `train.csv`, `validation_input.csv`, `forecast_index_validation.csv`, and `metadata.json` from the official Hugging Face repository:
```bash
python scripts/download_data.py
```

### Step 2: Run and Evaluate Baselines
Computes predictions for `naive_last_value`, `seasonal_mean`, `lag24`, and `lag168` and saves to `submissions/baselines/`:
```bash
python scripts/run_and_evaluate_baselines.py
```

### Step 3: Train the Top-Tier Ensemble (Winning Model)
Trains the multi-seed PyTorch `DeepOperationsNet` on CUDA and blends with multi-scale GBDT:
```bash
python scripts/train_top_ensemble.py
```
*Outputs: `submissions/ensemble_v2.csv` (Score: **14.33% WAPE**).*

### Step 4: Run Ablation Studies
Executes the full ablation suite (Architecture comparison, $\ell_1$ vs. Huber vs. MSE loss, and feature group importance):
```bash
python scripts/run_ablations.py
```
*Outputs saved to: `results/ablations/`.*

### Step 5: Calibrate Conformal Prediction Intervals
Generates distribution-free prediction bands with empirical coverage verification:
```bash
python scripts/run_conformal_prediction.py
```
*Outputs: `80% nominal -> 81.22% empirical coverage`, `90% nominal -> 90.19% empirical coverage`.*

### Step 6: Run Additional Dataset Generalization
Evaluates the PyTorch architecture on real-world Binance cryptocurrency trading volume across 12 major assets (Julien, 2023 via `kagglehub`):
```bash
python scripts/run_crypto_experiment.py
```
*Automatically downloads and caches real Binance hourly data (51,840 observations across BTC, ETH, BNB, SOL, ADA, DOGE, DOT, AVAX, ATOM, ALGO, AAVE, BCH, CRV), trains the deep network with learned asset embeddings on CUDA, and evaluates against naive and seasonal baselines.*
*Outputs saved to: `results/additional_dataset/crypto_experiment_results.csv`.*

---

## Additional Dataset Generalization Benchmark (Crypto Volume)

To test domain transferability beyond industrial operations, we evaluated our PyTorch architecture on real Binance hourly trading volume data ([Julien, 2023](https://www.kaggle.com/datasets/franoisgeorgesjulien/crypto)) over a 336-hour horizon across 12 major cryptocurrency pairs:

| Model Architecture | WAPE $\downarrow$ | MAE $\downarrow$ | MSE $\downarrow$ | RMSE $\downarrow$ | sMAPE (\%) $\downarrow$ | Performance vs. Baselines |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Last-Value Baseline** | `0.0534` | `0.7108` | `0.8324` | `0.9124` | `5.44` | Repeat last value baseline |
| **Seasonal Mean Baseline** | `0.0523` | `0.6963` | `0.7268` | `0.8525` | `5.31` | Diurnal cycle reference |
| **Our PyTorch DeepNet Architecture** | **`0.0312`** | **`0.4155`** | **`0.2732`** | **`0.5227`** | **`3.22`** | **>40% Error Reduction (62.4% MSE reduction)** |

> **Key Finding:** On real-world financial data, price volatility ($\frac{\text{High} - \text{Low}}{\text{Close}}$) and price returns create non-linear volume shocks that statistical baselines cannot anticipate. Our deep ResNet successfully captures these non-linear dependencies and leverages learned entity embeddings to share cross-asset liquidity dynamics.

---

## Final Private Model Evaluation Verification

The final evaluation is run privately by the course instructors using an offline container environment. We verified our submission package (`final_submission.zip`) against these requirements:

### Verification Command:
```bash
python student/submission_template/predict.py \
  --input_dir data \
  --output_file submissions/test_sim_pred.csv \
  --checkpoint student/submission_template/checkpoint.pt
```

### Compliance Checklist:
* [x] **Pure PyTorch Implementation**: Model is implemented in PyTorch (`ForecastModel`).
* [x] **Zero Internet Dependency**: Inference runs completely offline with self-contained weights and feature transformers.
* [x] **Fast Inference**: Produces all 32,256 required predictions in **< 1.0 second**.
* [x] **Exact Output Schema**: Generates `series_id,timestamp,prediction` matching `forecast_index_validation.csv` with 0 nulls.

---

## Group Contributions

In accordance with course guidelines, the project contributions were distributed as follows:
* **Adel Bukseisa**: Data acquisition pipeline, domain feature engineering, PyTorch `DeepOperationsNet` modeling, and Hugging Face leaderboard submissions.
* **Mariam Farhat**: Baseline implementations, loss function ablation experiments ($\ell_1$ vs. Huber vs. MSE), and hyperparameter optimization.
* **Mehshid Atiq**: Missingness robustness analysis, feature group importance ablations, and LaTeX report composition.
* **Omar Elmorsy**: Additional crypto dataset experiment, split conformal prediction uncertainty calibration, and literature review.

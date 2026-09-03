# Final Model Submission — Grex XXIII (Group 23)

**Course:** Deep Learning (SS26), Technical University of Darmstadt  
**Leaderboard Group:** Grex XXIII (Group ID: 23)  
**Winning Model Name:** `nuhhh_v2`  
**Public Validation Leaderboard Score:** **14.33% WAPE** (Rank #87)  
**GitHub Repository:** [https://github.com/TipXI/DLAM-Multivariate-Time-Series-Forecasting-2026](https://github.com/TipXI/DLAM-Multivariate-Time-Series-Forecasting-2026)  

---

## 1. Package Contents

This archive (`final_submission.zip`) contains all necessary files for offline private test evaluation:
* `predict.py`: Self-contained offline inference entry point.
* `checkpoint.pt`: PyTorch `ForecastModel` weights and feature normalizers.
* `src/model.py`: PyTorch `ForecastModel` architecture definition (`DeepOperationsNet`).
* `requirements.txt`: Minimal runtime dependencies (`torch`, `pandas`, `numpy`).
* `README.md`: Reproduction and inference instructions.

---

## 2. Official Evaluation Command (Lecturer Evaluation)

During private evaluation, instructors run the exact command:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

### Argument Specifications:
* `--input_dir`: Directory containing `test_input.csv` (or `validation_input.csv`) and `forecast_index_test.csv` (or `forecast_index_validation.csv`).
* `--output_file`: Path where the output prediction CSV will be written.
* `--checkpoint`: Path to the PyTorch checkpoint (`checkpoint.pt` or `/submission/checkpoint.pt`).

The output CSV strictly adheres to the required competition schema:
```csv
series_id,timestamp,prediction
```
Execution time is approximately **0.8 seconds** for all 32,256 predictions with zero null values and zero internet dependency.

---

## 3. Environment & Dependencies

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

* `torch>=2.2.0`
* `pandas>=2.2.0`
* `numpy>=1.26.0`

---

## 4. How to Reproduce Full Training

To reproduce full model training from scratch using the raw benchmark dataset:
1. Clone our repository:
   ```bash
   git clone https://github.com/TipXI/DLAM-Multivariate-Time-Series-Forecasting-2026.git
   cd DLAM-Multivariate-Time-Series-Forecasting-2026
   pip install -r requirements.txt
   ```
2. Download data:
   ```bash
   python scripts/download_data.py
   ```
3. Train the top-tier ensemble:
   ```bash
   python scripts/train_top_ensemble.py
   ```
   This generates `submissions/ensemble_v2.csv` (our winning submission achieving **14.33% WAPE** on the public leaderboard).

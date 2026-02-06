# Experiments Zone

This directory contains experiment runners for systematic model evaluation.

## Available Experiments

| Script | Description | Runtime |
|--------|-------------|---------|
| `run_ablation.py` | Full factorial ablation study | ~hours |
| `run_cross_validation.py` | K-fold cross-validation | ~hours |
| `run_snr_robustness.py` | Noise robustness testing | ~30min |
| `run_baselines.py` | Baseline model comparison | ~hours |

## Quick Start

### Run All Experiments
```bash
python run_all_experiments.py
```

### Quick Test Mode (2 epochs)
```bash
python run_all_experiments.py --quick
```

### Individual Experiments
```bash
python experiments/run_ablation.py --quick
python experiments/run_cross_validation.py
python experiments/run_snr_robustness.py
python experiments/run_baselines.py
```

## Output

All results are saved to `results/`:
- `results/ablation_*.csv` - Ablation study results
- `results/cross_validation/` - CV fold results
- `results/snr_robustness/` - SNR test results
- `results/figures/` - Generated plots

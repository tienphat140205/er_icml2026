# Stage 2 Router Training

## Quick Start

```bash
cd /path/to/er_icml2026

# Step 1: Setup environment (first time only)
bash env.sh

# Step 2: Run hyperparameter sweep
bash verl/stage2_scripts/trainer/run_sweep.sh
```

## Hyperparameter Sweep

The sweep runs 5 configurations sequentially:

| Phase | BETA_MAX | BETA_MIN | RHO | LR | Description |
|-------|----------|----------|-----|-----|-------------|
| 1 | 1.0 | 0.5 | 0.7 | 5e-5 | Baseline |
| 2 | 1.0 | 0.5 | 0.8 | 5e-5 | More reward mixing |
| 3 | 0.8 | 0.3 | 0.7 | 5e-5 | Lower temps, sharper |
| 3 | 0.5 | 0.5 | 0.7 | 5e-5 | Constant temp (no annealing) |
| 4 | 1.0 | 0.5 | 0.7 | 1e-5 | Lower LR |

## Files

- `run_sweep.sh` - Hyperparameter sweep runner
- `run_3b.sh` - Single experiment runner (called by sweep)

# Stage 2 Router Training

## Quick Start

```bash
cd /path/to/er_icml2026

# Step 1: Setup environment (first time only)
conda env create -f environment/verl_env.yaml
conda activate arm_verl
pip3 install --force-reinstall torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install --upgrade math-verify

# Step 2: Run hyperparameter sweep
bash verl/stage2_scripts/trainer/run_sweep.sh
```

## Docker Setup (Alternative)

```bash
cd /path/to/er_icml2026

# Build image
docker build -t arm-train .

# Run training
docker run --gpus all -it --shm-size=16g \
  -v $(pwd):/workspace \
  arm-train bash verl/stage2_scripts/trainer/run_sweep.sh
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

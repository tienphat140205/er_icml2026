#!/bin/bash
# Smart Hyperparameter Sweep
# Baseline LR: 5e-5

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

configs=(
    # ========== Phase 1: Baseline ==========
    "1.0 0.5 0.7 5e-5"   # baseline
    
    # ========== Phase 2: Vary RHO (reward mixing) ==========
    "1.0 0.5 0.8 5e-5"   # more reward mixing
    
    # ========== Phase 3: Vary BETA (temp annealing) ==========
    "0.8 0.3 0.7 5e-5"   # lower temps, sharper
    "0.5 0.5 0.7 5e-5"   # constant temp (no annealing)
    
    # ========== Phase 4: Vary LR ==========
    "1.0 0.5 0.7 1e-5"   # lower lr
)

echo "Total configs: ${#configs[@]}"

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    read ROUTER_BETA_MAX ROUTER_BETA_MIN ROUTER_RHO ROUTER_LR <<< "$config"
    export ROUTER_BETA_MAX ROUTER_BETA_MIN ROUTER_RHO ROUTER_LR
    
    echo "========================================"
    echo "[$((i+1))/${#configs[@]}] beta_max=$ROUTER_BETA_MAX beta_min=$ROUTER_BETA_MIN rho=$ROUTER_RHO lr=$ROUTER_LR"
    echo "========================================"
    
    "$SCRIPT_DIR/run_3b.sh"
done

echo "All experiments completed!"

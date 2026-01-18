#!/bin/bash
# Download Miniconda
sudo apt update -y
sudo apt install tmux -y

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install to /workspace/miniconda
bash miniconda.sh -b -p /workspace/miniconda

# Initialize conda
/workspace/miniconda/bin/conda init bash

# Cleanup
rm miniconda.sh

echo "Installation complete. Please restart your shell or run 'source ~/.bashrc' to activate conda."


/workspace/miniconda/bin/conda env create -f arm/environment/verl_env.yaml -y

source /workspace/miniconda/bin/activate
conda activate arm_verl

pip3 install --force-reinstall torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

pip3 install --upgrade math-verify



/workspace/miniconda/bin/conda env create -f environment/llama_factory_env.yaml -y
source /workspace/miniconda/bin/activate
conda activate arm_llama_factory
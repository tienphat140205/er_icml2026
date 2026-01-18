# ARM Router Training - Stage 2
FROM continuumio/miniconda3:latest

WORKDIR /workspace

# Copy environment file and verl package
COPY environment/verl_env.yaml /workspace/environment/verl_env.yaml
COPY verl/ /workspace/verl/

# Create conda environment from yaml
RUN conda env create -f environment/verl_env.yaml

# Activate environment and install remaining packages
SHELL ["conda", "run", "-n", "arm_verl", "/bin/bash", "-c"]

# Force reinstall torch with CUDA 12.4
RUN pip3 install --force-reinstall torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn
RUN pip3 install flash-attn --no-build-isolation

# Install math-verify
RUN pip3 install --upgrade math-verify

# Set environment variables
ENV WANDB_MODE=offline
ENV RAY_memory_usage_threshold=0.99
ENV VLLM_ATTENTION_BACKEND=XFORMERS

# Make conda env active by default
RUN echo "conda activate arm_verl" >> ~/.bashrc
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arm_verl"]
CMD ["bash"]

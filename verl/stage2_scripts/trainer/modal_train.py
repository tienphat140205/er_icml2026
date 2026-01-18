import modal
import os

app = modal.App("arm-verl-train-v2")  # New app name to avoid cache

# Use standard Nvidia CUDA base image - most reliable approach
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install("git", "wget", "curl", "build-essential")
    
    # Install other Python dependencies (comprehensive from verl_env.yaml)
    .pip_install(
        # Core ML
        "vllm==0.6.3",
        "transformers==4.49.0",
        "ray==2.43.0",
        "wandb==0.19.8",
        "accelerate==1.5.2",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "numpy==1.26.4",
        "pandas",
        "datasets",
        "peft",
        "trl",
        "huggingface-hub",
        "pydantic",
        "tensorboard",
        "xformers==0.0.27.post2",
        "triton==3.0.0",
        # Serialization
        "cloudpickle",
        "dill",
        "msgpack",
        "pyzmq",
        # Utils
        "lark",
        "codetiming",
        "sentencepiece",
        "tiktoken",
        "einops",
        "bitsandbytes",
        "safetensors",
        "pyarrow",
        "pyyaml",
        "python-dotenv",
        "tqdm",
        "regex",
        "requests",
        # Additional verified missing
        "distro",
        "httptools",
        "msgspec",
        "openai",
        "opencv-python-headless",
        # vLLM dependencies
        "outlines",
        "compressed-tensors",
        "fastapi",
        "uvicorn",
        "uvloop",
        "nvitop",
        # math-verify dependencies
        "sympy",
        "latex2sympy2-extended",
        "antlr4-python3-runtime==4.9.3",
        # verl core
        "tensordict",
    )
    
    # Post-install steps (exactly like env.sh)
    .run_commands(
        "pip install wheel setuptools",
        # 1. Force reinstall torch
        "pip3 install --force-reinstall torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
        # 2. Flash-attn
        "pip3 install flash-attn --no-build-isolation",
        # 3. Math-verify upgrade
        "pip3 install --upgrade math-verify"
    )
    .run_commands(
        # HF fast transfer
        "pip install hf_transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "NCCL_DEBUG": "INFO",
        "PYTHONPATH": "/root/arm/verl:$PYTHONPATH"
    })
    
    # Copy project code
    .add_local_dir(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
        remote_path="/root/arm",
        ignore=["*.git", "__pycache__", "checkpoints", "*.pt", "*.bin", "*.safetensors"]
    )
)

# Volumes
from modal import Volume
vol_hf_cache = Volume.from_name("hf-cache", create_if_missing=True)
vol_checkpoints = Volume.from_name("verl-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100:4",
    timeout=86400,
    volumes={
        "/root/.cache/huggingface": vol_hf_cache,
        "/root/arm/checkpoints": vol_checkpoints
    }
)
def train_main():
    import os
    import subprocess
    import sys

    print("Starting training on Modal...")
    os.chdir("/root/arm")
    
    print("Python:", sys.executable)
    subprocess.run(["nvidia-smi"], check=True)

    script_path = "verl/stage2_scripts/trainer/run_3b.sh"
    subprocess.run(["chmod", "+x", script_path], check=True)
    
    print(f"Executing {script_path}...")
    
    process = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
        
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, script_path)

if __name__ == "__main__":
    modal.enable_output()
    with app.run():
        train_main.remote()

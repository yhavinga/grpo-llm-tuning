# Multi-Platform GRPO Tuning Scripts

This repository contains scripts and workflows for setting up and running Group Relative Policy Optimization (GRPO) fine-tuning experiments for Large Language Models (LLMs) on various hardware accelerators.

Currently, it includes support for Google Cloud TPUs using the [EasyDeL](https://github.com/erfanzar/easydel) framework.

**Future Goals:**

*   Add support for NVIDIA GPUs using frameworks like [TRL](https://github.com/huggingface/trl). (Partially Implemented - See CUDA section)
*   Add support for AMD MI300X GPUs (likely using TRL or similar ROCm-compatible frameworks).

## Current TPU Workflow (EasyDeL)

The current scripts streamline the process of creating a TPU VM, installing dependencies, and running a test GRPO job using EasyDeL.

### Prerequisites

1.  **Google Cloud SDK (`gcloud`)**: Install and configure the `gcloud` CLI.
    *   Installation: [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install)
    *   Configuration: Run `gcloud init` and `gcloud auth login`.
    *   Enable APIs: Ensure the Compute Engine API and Cloud TPU API are enabled for your project:
        ```bash
        gcloud services enable compute.googleapis.com tpu.googleapis.com --project=YOUR_PROJECT_ID
        ```
    *   Application Default Credentials (ADC): Set up ADC for tools like `eopod`:
        ```bash
        gcloud auth application-default login
        ```

2.  **Python Environment**: A Python 3 environment (e.g., 3.10 or later) is recommended.

3.  **Local Tools (`eopod`)**: Install `eopod` for interacting with TPU VMs.

### Local Setup

It's highly recommended to use a virtual environment:

```bash
# 1. Create a virtual environment (if you don't have one)
python -m venv venv  # Or use python3

# 2. Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (Git Bash/WSL):
# source venv/Scripts/activate
# On Windows (CMD/PowerShell):
# venv\Scripts\activate.bat  OR  venv\Scripts\Activate.ps1

# 3. Install eopod
pip install eopod

# 4. (Optional) Verify gcloud configuration
gcloud config list
gcloud auth list
```

### Configuration (`tpu/.env`)

Before running the scripts, copy the example environment file and fill in your specific details:

```bash
cp tpu/.env.example tpu/.env
# Now edit tpu/.env with your text editor
```

Key variables in `tpu/.env`:

*   `GCP_PROJECT`: Your Google Cloud Project ID.
*   `TPU_NAME`: The desired name for your TPU VM.
*   `ZONE`: The GCP zone where the TPU will be created (e.g., `us-central2-b`).
*   `ACCELERATOR_TYPE`: The type of TPU (e.g., `v4-8`).
*   `RUNTIME_VERSION`: The TPU VM runtime (e.g., `tpu-ubuntu2204-base`).
*   `EMAIL` (Optional): Email for notifications.
*   `HF_TOKEN` (Optional): Hugging Face token for private models/datasets.
*   `WANDB_TOKEN` (Optional): Weights & Biases API key for logging.

### Usage

Run the scripts in sequence from the root directory of the repository:

1.  **Create TPU VM:**
    ```bash
    bash tpu/01_create_vm.sh
    ```
    This script will request a queued TPU resource and wait until it's allocated and provisioned. It might take some time depending on resource availability.

2.  **Install Dependencies on TPU VM:**
    ```bash
    bash tpu/02_install_with_eopod.sh
    ```
    This uses `eopod` to install EasyDeL, JAX (with TPU support), and other required libraries on the remote TPU VM. It also handles optional Hugging Face/WandB logins if tokens are provided in `.env`.

3.  **Run GRPO Test Job:**
    ```bash
    bash tpu/03_grpo_test_run.sh
    ```
    This launches a short test GRPO fine-tuning job on the TPU VM using the `gsm8k_grpo` script from EasyDeL, a small Llama 3 model, and settings configured for a quick check. Monitor the terminal output (or WandB if enabled).

### Included Files (`tpu/` directory)

*   `01_create_vm.sh`: Creates/waits for TPU VM.
*   `02_install_with_eopod.sh`: Installs dependencies on the VM.
*   `03_grpo_test_run.sh`: Runs a sample GRPO training job.
*   `grpo_explained.md`: Detailed explanation of GRPO concepts within EasyDeL.
*   `.env.example`: Template for the required environment variables.

## CUDA/GPU Workflow (TRL - WIP)

This section outlines the setup for running GRPO fine-tuning on NVIDIA GPUs. It currently focuses on dependency installation.

### Prerequisites

1.  **NVIDIA Driver**: Ensure you have a compatible NVIDIA driver installed. [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2.  **CUDA Toolkit**: Install the CUDA Toolkit (version 12.1 or compatible with your driver and PyTorch build). [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
3.  **Python Environment**: A Python 3 environment (e.g., 3.10 or later) is recommended. Use a virtual environment.
4.  **`pip`**: Ensure `pip` is up-to-date (`pip install --upgrade pip`).

### Local Setup

It's highly recommended to use a virtual environment:

```bash
# 1. Create a virtual environment (if you don't have one)
python -m venv venv  # Or use python3

# 2. Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (Git Bash/WSL):
# source venv/Scripts/activate
# On Windows (CMD/PowerShell):
# venv\Scripts\activate.bat  OR  venv\Scripts\Activate.ps1

# 3. Ensure pip is updated
pip install --upgrade pip
```

### Usage

Run the installation script from the root directory:

1.  **Install Dependencies:**
    ```bash
    bash cuda/10_cuda_install.sh
    ```
    This script installs PyTorch (pinned to 2.5.1 for CUDA 12.1), `flash-attn` (pinned to 2.7.4.post1 for pre-built wheels), `trl`, `transformers`, and other necessary libraries.

*(Note: Training scripts using these dependencies are under development.)*

### Included Files (`cuda/` directory)

*   `10_cuda_install.sh`: Installs dependencies for the CUDA environment.
*   `grpo_finetune_trl.py`: (Placeholder/WIP) Script for running the actual fine-tuning.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for adding support for new hardware, frameworks, or improving existing scripts. 
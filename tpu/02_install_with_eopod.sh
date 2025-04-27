#!/bin/bash
set -eo pipefail # Exit immediately if a command exits with a non-zero status or if any command in a pipeline fails.

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/.env"

if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment variables from ${ENV_FILE}"
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "Error: Environment file ${ENV_FILE} not found."
    echo "Please create it based on .env.example or ensure 01_create_vm.sh was run successfully."
    exit 1
fi

# --- Validate Required Variables for eopod ---
# We mainly need project, zone, and tpu name for configuration
required_vars=("GCP_PROJECT" "TPU_NAME" "ZONE")
missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "Error: Missing required environment variables in ${ENV_FILE} for eopod configuration:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# --- Ensure eopod is installed locally ---
echo "Checking for local eopod installation..."
if ! command -v eopod &> /dev/null
then
    echo "eopod command could not be found locally."
    echo "Please install it first using: pip install eopod"
    # Optionally add local bin to PATH if needed
    # echo 'Consider adding ~/.local/bin to your PATH:'
    # echo 'export PATH="$HOME/.local/bin:$PATH"'
    exit 1
fi
echo "eopod found."

# --- 1. Configure eopod ---
echo "Configuring eopod for TPU: ${TPU_NAME}..."
eopod configure --project-id "${GCP_PROJECT}" --zone "${ZONE}" --tpu-name "${TPU_NAME}"

# --- 2. Install Core Dependencies on TPU VM ---
echo "Installing core dependencies (tensorflow, tensorflow-datasets, torch) on TPU VM via eopod..."
eopod run pip install tensorflow tensorflow-datasets
eopod run pip install torch --index-url https://download.pytorch.org/whl/cpu

# Add JAX TPU installation
echo "Installing JAX with TPU support on TPU VM via eopod..."
eopod run "pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# --- 3. Install EasyDeL on TPU VM ---
echo "Installing EasyDeL from GitHub on TPU VM via eopod..."
# Using --upgrade to ensure the latest version is pulled if it was previously installed
eopod run pip install --upgrade git+https://github.com/erfanzar/easydel

# --- 4. (Optional) Authentication Setup --- 
echo "Authentication steps (Hugging Face, WandB) are optional."
# Check for environment variables to automate login if desired

# Hugging Face Login
if [ -n "${HF_TOKEN}" ]; then
    echo "Attempting Hugging Face login using HF_TOKEN environment variable..."
    eopod run "python -c 'from huggingface_hub import login; login(token=\"${HF_TOKEN}\")'"
elif [ -n "${HF_LOGIN_INTERACTIVE}" ] && [ "${HF_LOGIN_INTERACTIVE}" = "true" ]; then
    echo "Please follow the prompts to log in to Hugging Face Hub on the TPU VM..."
    eopod run "python -c 'from huggingface_hub import login; login()'"
else
    echo "Skipping Hugging Face login. Set HF_TOKEN or HF_LOGIN_INTERACTIVE=true in .env to enable."
fi

# WandB Login
if [ -n "${WANDB_TOKEN}" ]; then
    echo "Attempting Weights & Biases login using WANDB_TOKEN environment variable..."
    eopod run python -m wandb login "${WANDB_TOKEN}"
elif [ -n "${WANDB_LOGIN_INTERACTIVE}" ] && [ "${WANDB_LOGIN_INTERACTIVE}" = "true" ]; then
    echo "Please follow the prompts to log in to Weights & Biases on the TPU VM..."
    eopod run python -m wandb login
else
    echo "Skipping Weights & Biases login. Set WANDB_TOKEN or WANDB_LOGIN_INTERACTIVE=true in .env to enable."
fi


echo "--- Installation Script Completed ---"
echo "EasyDeL and dependencies should now be installed on ${TPU_NAME}."
echo "You can verify by SSHing into the TPU and checking installed packages or running EasyDeL commands." 
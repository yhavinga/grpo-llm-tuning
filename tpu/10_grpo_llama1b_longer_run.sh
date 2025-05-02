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

# --- Validate Required Variables ---
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

# --- Ensure eopod is installed locally (Quick Check) ---
if ! command -v eopod &> /dev/null; then
    echo "eopod command could not be found locally. Please install it: pip install eopod"
    exit 1
fi

# --- 1. Configure eopod (Idempotent) ---
# Ensures eopod targets the correct TPU defined in .env
echo "Configuring eopod for TPU: ${TPU_NAME} in zone ${ZONE}..."
eopod configure --project-id "${GCP_PROJECT}" --zone "${ZONE}" --tpu-name "${TPU_NAME}"

# --- 2. Define WandB Usage ---
WANDB_ARGS=""
if [ -n "${WANDB_TOKEN}" ]; then
    echo "WANDB_TOKEN found in .env - Enabling Weights & Biases logging."
    WANDB_ARGS="--use_wandb"
    # Note: Assumes wandb login was handled by 02_install_with_eopod.sh or manually
else
    echo "WANDB_TOKEN not found in .env - Skipping Weights & Biases logging."
    echo "Set WANDB_TOKEN in .env to enable."
fi

# --- 3. Construct and Run the GRPO Longer Run Command ---
echo "Constructing the GRPO longer run command..."

# Construct the command as a single string to avoid parsing issues with escaped newlines
# Note: This uses the gsm8k_grpo script which is tailored for the GSM8K dataset.
# For custom datasets, use the GRPOTrainer class in a Python script.
# Parameters here are set for a *longer training run* on Llama-3.2-1B.

CMD="python -m easydel.scripts.finetune.gsm8k_grpo"   # The EasyDeL module to execute.
CMD+=" --repo_id meta-llama/Llama-3.2-1B-Instruct" # Using Llama-3.2-1B for a faster longer run than 8B.
CMD+=" --attn_mechanism 'vanilla'"                # Changed to 'vanilla' based on documentation example and head count mismatch error.
                                                # 'flash_attn2' caused: `ValueError: Head count mismatch: got 8, 32, 32`
CMD+=" --sharding_axis '1,1,-1,1'"                # ** Sharding Strategy **
                                                # Specifies how model parameters/activations are sharded across TPU axes.
                                                # Format: (dp, fsdp, tp, sp). -1 fills available devices.
                                                # Common Choices & Issues Seen:
                                                #   '1,-1,1,1': Biases towards FSDP (dp=1, fsdp=N, tp=1, sp=1).
                                                #               -> Used to fix Gemma-2B KV Cache error (`ValueError: ... dimension 2 should be divisible by 4, but it is equal to 1`).
                                                #               -> Caused error with Llama-3-8B in `compute_refmodel_logps` (`ValueError: Sharding passed to pjit does not match...`).
                                                #   '1,1,-1,1': Biases towards Tensor Parallelism (dp=1, fsdp=1, tp=N, sp=1).
                                                #               -> Caused KV Cache head dimension error with Gemma-2B (MQA/GQA).
                                                #               -> Seems required for Llama-3-8B/1B `compute_refmodel_logps`.
                                                # ** Conclusion: May need adjustment based on model architecture (MQA/GQA) and specific errors encountered during pjit compilation. **
CMD+=" --max_prompt_length 512"                   # Maximum token length for input prompts.
CMD+=" --max_completion_length 256"               # Maximum token length for generated completions.
CMD+=" --beta 0.04"                               # GRPO hyperparameter, controlling the KL divergence penalty (policy deviation).
                                                # Start low (0.01-0.05) and tune.
CMD+=" --top_p 0.95"                              # Nucleus sampling probability for generation (used by vInference).
CMD+=" --top_k 50"                                # Top-k sampling limit for generation.
CMD+=" --num_return_sequences 4"                  # Number of candidate completions to generate per prompt for comparison.
                                                # More sequences provide richer signal but increase memory/compute.
# --- Reward parameters specific to the gsm8k_grpo script's heuristic ---
CMD+=" --xml_reward 0.125"                        # Heuristic reward component (likely related to structured output).
CMD+=" --xml_full_match_reward 0.5"               # Heuristic reward component.
CMD+=" --xml_full_match_reject 0.0"               # Heuristic reward component.
CMD+=" --correctness_reward 2.0"                  # Heuristic reward component (likely for matching the final answer).
# --- End of gsm8k_grpo specific reward parameters ---
CMD+=" --total_batch_size 8"                      # Total effective batch size across all devices (adjust based on memory).
CMD+=" --learning_rate 1e-5"                      # Initial learning rate for the optimizer.
CMD+=" --learning_rate_end 1e-6"                  # Final learning rate for the scheduler (if using linear/cosine).
CMD+=" --log_steps 50"                            # Log metrics every N steps (increased for longer run).
CMD+=" --num_train_epochs 3"                      # Number of epochs for the longer run.
CMD+=" --save_steps 500"                          # Save checkpoint every N steps (increased for longer run).
CMD+=" --shuffle_train_dataset true"              # Shuffle the training dataset each epoch.
CMD+=" --report_steps 1"                          # Report metrics (e.g., WandB) every N steps.
CMD+=" --progress_bar_type 'tqdm'"                # Progress bar style ('tqdm', 'rich', 'json').
CMD+=" --auto_shard_states true"                  # Automatically shard optimizer states across devices (recommended).
CMD+=" --optimizer 'adamw'"                       # Optimizer to use ('adamw', 'lion', 'adafactor', etc.).
CMD+=" --scheduler 'linear'"                      # Learning rate scheduler type ('linear', 'cosine', 'none').
CMD+=" --do_last_save true"                       # Ensure a final checkpoint is saved upon completion/interruption.
CMD+=" --dtype 'bfloat16'"                        # Data type for computation (bfloat16 is efficient on TPUs).
CMD+=" --param_dtype 'bfloat16'"                    # Data type for model parameters.
CMD+=" ${WANDB_ARGS}"                              # Append '--use_wandb' if WANDB_TOKEN is set.

# Execute the command string via eopod
echo "Executing command on TPU VM:"
echo "${CMD}"
eopod run "${CMD}"

# --- 4. Completion Message ---
echo ""
echo "--- GRPO Longer Run Command Submitted ---"
echo "The script has submitted the training job to the TPU VM (${TPU_NAME}) via eopod."
echo "Monitor the output in your terminal or check WandB if enabled."
echo "This is configured for a longer run based on epochs." 
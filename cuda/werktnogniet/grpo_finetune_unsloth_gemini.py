# --- Dependencies ---
# Please ensure you have the following packages installed:
# pip install unsloth datasets transformers torch accelerate trl wandb pillow vllm
# Note: vLLM is used by Unsloth for fast inference if `fast_inference=True`.

from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams # Used for model.fast_generate
import wandb

# --- Configuration ---
BASE_MODEL_ID = "google/gemma-3-1b-it"
DATASET_ID = "openai/gsm8k"
NEW_MODEL_NAME_BASE = "Gemma3-1B-GRPO-Unsloth-GSM8K"
WANDB_PROJECT = "Unsloth-GRPO-GSM8K"

# Unsloth specific
MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
LORA_RANK = 32        # Larger rank = smarter, but slower. Suggested 8, 16, 32, 64, 128

# GRPO specific
MAX_PROMPT_LENGTH = 512 # Max length for prompts
# max_completion_length will be MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

# --- Weights & Biases Login ---
try:
    wandb.login()
    WANDB_ENABLED = True
    print("Successfully logged in to W&B.")
except Exception as e:
    print(f"Could not log in to W&B: {e}")
    print("Proceeding without W&B logging.")
    WANDB_ENABLED = False

# --- Load Model and Tokenizer with Unsloth ---
print(f"Loading model using Unsloth: {BASE_MODEL_ID}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,      # Use 4-bit quantization
    fast_inference=False,   # Disable vLLM fast inference to avoid vllm_engine attribute error
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.8, # Adjust if out of memory
    # token="hf_...", # Optional: use if model is private
)
print("Base model and tokenizer loaded with Unsloth.")

# --- Configure PEFT Model with Unsloth ---
print("Configuring PEFT model with Unsloth...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth", # Enable long context finetuning
    random_state=3407,
)
print("PEFT model configured with Unsloth.")
model.print_trainable_parameters()

# Set pad token if not already set (Unsloth might handle this, but good practice)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Note: Unsloth's FastLanguageModel might handle pad_token_id internally.
    # If issues arise, ensure model.config.pad_token_id is also set.
    print("Pad token set to EOS token.")


# --- Data Preparation ---
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Helper functions to extract answers
def extract_xml_answer(text: str) -> str:
    parts = text.split("<answer>")
    if len(parts) > 1:
        answer = parts[-1].split("</answer>")[0]
        return answer.strip()
    return "" # Return empty if format is not met

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Function to prepare the GSM8K dataset
def get_gsm8k_questions(split="train") -> Dataset:
    print(f"Loading dataset: {DATASET_ID} ({split} split)...")
    data = load_dataset(DATASET_ID, "main", split=split)
    print("Dataset loaded, preparing prompts...")

    formatted_data = []
    for example in data:
        # Format the system and user messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        
        # Extract the answer
        answer = extract_hash_answer(example["answer"]) or ""
        
        # Create a single example dict
        formatted_example = {
            "prompt": messages,  # Store as messages format for GRPOTrainer
            "answer": answer,
            "answer_raw": example["answer"]  # Keep the raw answer for reference
        }
        
        formatted_data.append(formatted_example)
    
    # Create a Dataset from the formatted data
    formatted_dataset = Dataset.from_list(formatted_data)
    print(f"Dataset prepared with {len(formatted_dataset)} examples")
    
    # Print an example for debugging
    print("\nExample prompt:")
    for msg in formatted_dataset[0]["prompt"]:
        print(f"{msg['role']}: {msg['content']}")
    print(f"Example answer: {formatted_dataset[0]['answer']}")
    
    return formatted_dataset

train_dataset = get_gsm8k_questions("train")
# test_dataset = get_gsm8k_questions("test") # Optionally load test set

# --- Defining Reward Functions ---
# Reward function that checks if the answer is correct
def correctness_reward_func(prompts, completions, answer_raw, answer, **kwargs) -> list[float]:
    # `completions` is a list of lists of dicts. Each inner list is for one prompt.
    # Each dict has "role" and "content". We are interested in the assistant's reply.
    # Example: completions[prompt_index][generation_index]['content']
    
    rewards = []
    for i in range(len(prompts)): # Iterate over batch
        batch_rewards = []
        # `answer` here is the reference answer from the dataset for the i-th prompt
        actual_answer_str = answer[i]
        for gen_idx in range(len(completions[i])): # Iterate over generations for this prompt
            response_content = completions[i][gen_idx]["content"]
            extracted_response = extract_xml_answer(response_content)
            
            if i == 0 and gen_idx == 0: # Print for the first prompt, first generation for debugging
                 print("-" * 20)
                 print(f"Prompt:\n{prompts[i]}")
                 print(f"Reference Answer:\n{actual_answer_str}")
                 print(f"Generated Response:\n{response_content}")
                 print(f"Extracted Response:\n{extracted_response}")
            
            # Ensure actual_answer_str is not None before comparison
            if actual_answer_str is not None and extracted_response == actual_answer_str:
                batch_rewards.append(2.0)
            else:
                batch_rewards.append(0.0)
        rewards.append(batch_rewards)
    return rewards


# Reward function that checks if the answer is an integer
def int_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for i in range(len(completions)): # Iterate over batch
        batch_rewards = []
        for gen_idx in range(len(completions[i])):
            response_content = completions[i][gen_idx]["content"]
            extracted_response = extract_xml_answer(response_content)
            if extracted_response.isdigit() or (extracted_response.startswith('-') and extracted_response[1:].isdigit()):
                batch_rewards.append(0.5)
            else:
                batch_rewards.append(0.0)
        rewards.append(batch_rewards)
    return rewards

# Reward function that checks if the completion follows the strict format
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$" # Allow trailing whitespace
    rewards = []
    for i in range(len(completions)):
        batch_rewards = []
        for gen_idx in range(len(completions[i])):
            response_content = completions[i][gen_idx]["content"]
            if re.match(pattern, response_content, re.DOTALL): # re.DOTALL allows '.' to match newlines
                batch_rewards.append(0.5)
            else:
                batch_rewards.append(0.0)
        rewards.append(batch_rewards)
    return rewards

# Reward function that checks if the completion follows a more relaxed format
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>" # DOTALL for content, \s* for flexible spacing
    rewards = []
    for i in range(len(completions)):
        batch_rewards = []
        for gen_idx in range(len(completions[i])):
            response_content = completions[i][gen_idx]["content"]
            if re.search(pattern, response_content, re.DOTALL): # Use re.search for substring match
                batch_rewards.append(0.5)
            else:
                batch_rewards.append(0.0)
        rewards.append(batch_rewards)
    return rewards

# Reward function that counts XML tags and penalizes extra content
def count_xml(text) -> float:
    score = 0.0
    # Ensure DOTALL is not implicitly assumed if not matching across newlines as intended
    # Original logic counts specific newline patterns.
    if text.count("<reasoning>\n") == 1: score += 0.125
    if text.count("\n</reasoning>\n") == 1: score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
        # Penalize content after the final intended structure
        extra_content_after_answer_tag = text.split("\n</answer>\n")
        if len(extra_content_after_answer_tag) > 1 and extra_content_after_answer_tag[-1].strip():
            score -= len(extra_content_after_answer_tag[-1].strip()) * 0.01 # Increased penalty factor
    if text.count("\n</answer>") == 1: # Could be \n</answer>\n or just \n</answer>
        score += 0.125
        extra_content_after_answer = text.split("\n</answer>")
        if len(extra_content_after_answer) > 1 and (extra_content_after_answer[-1].strip() and extra_content_after_answer[-1].strip() != "\n"):
             score -= (len(extra_content_after_answer[-1].strip())) * 0.01 # Increased penalty factor

    # Penalize if tags are missing
    if not ("<reasoning>" in text and "</reasoning>" in text and \
            "<answer>" in text and "</answer>" in text):
        score -= 0.5 # Significant penalty for malformed structure
    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for i in range(len(completions)):
        batch_rewards = []
        for gen_idx in range(len(completions[i])):
            response_content = completions[i][gen_idx]["content"]
            batch_rewards.append(count_xml(response_content))
        rewards.append(batch_rewards)
    return rewards

print("Reward functions defined.")

# --- Training with GRPO ---
print("Defining GRPO training arguments...")
training_args = GRPOConfig(
    output_dir=f"./{NEW_MODEL_NAME_BASE}_outputs",
    learning_rate=5e-6, # From example
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", # Unsloth supports paged optimizers
    logging_steps=1,
    per_device_train_batch_size=1, # As in example, adjust based on VRAM
    gradient_accumulation_steps=4, # Increased from example for potentially smoother training
    num_generations=6,             # Number of completions per prompt, from example
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
    num_train_epochs=1, # Set to 1 for a full training run, or use max_steps
    # max_steps=250,      # From example, for a shorter run
    save_steps=250,     # Save checkpoint every N steps
    max_grad_norm=0.1,  # From example
    report_to=["wandb"] if WANDB_ENABLED else "none",
    remove_unused_columns=False, # Important for GRPO custom reward signals
    bf16=not torch.cuda.is_bf16_supported(), # Use BF16 if supported, else True might default to FP16
    fp16=torch.cuda.is_bf16_supported(),   # Use FP16 if BF16 is not supported
    use_vllm=True,      # Still needed for GRPO with TRL 0.15.2+ even when fast_inference=False
                        # Helps resolve "list indices must be integers or slices, not str" error
)
if torch.cuda.is_bf16_supported():
    print("BF16 is supported. Using BF16 for training.")
    training_args.bf16 = True
    training_args.fp16 = False
else:
    print("BF16 not supported. Using FP16 for training.")
    training_args.bf16 = False
    training_args.fp16 = True


print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,  # Explicitly pass tokenizer to avoid processing issues
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
)
print("GRPOTrainer initialized.")

# --- Start Training ---
print("Starting GRPO training...")
if WANDB_ENABLED:
    try:
        wandb.init(project=WANDB_PROJECT, name=NEW_MODEL_NAME_BASE, config=training_args.to_dict())
        print(f"W&B run initialized for project: {WANDB_PROJECT}, run: {NEW_MODEL_NAME_BASE}")
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}. Continuing training without W&B.")
        trainer.args.report_to = "none"

trainer.train()
print("Training finished.")

# --- Save Final LoRA Adapters ---
final_lora_path = f"./{NEW_MODEL_NAME_BASE}_final_lora"
model.save_lora(final_lora_path)
print(f"Final LoRA adapters saved to {final_lora_path}")
tokenizer.save_pretrained(final_lora_path) # Save tokenizer alongside
print(f"Tokenizer saved to {final_lora_path}")


# --- Testing the Model (after training) ---
print("\n--- Testing the Fine-tuned Model ---")
# For Unsloth, if you used `fast_inference=True` and have vLLM,
# `model.fast_generate` is available.
# Ensure the LoRA adapter is loaded if it's not automatically applied after training.
# Unsloth's `get_peft_model` modifies the model in place, so it should have LoRA layers.

# To load a specific LoRA adapter for generation (if not already active or for a fresh model):
# model.load_lora(final_lora_path) # Or use model.load_lora("grpo_saved_lora") from example if that's the one

test_prompt_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Natalia sold clips to 48 of her friends and then found some more clips, so she sold half of the total clips to her brother. If she sold 20 clips to her brother, how many clips did she find?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(
    temperature=0.7, # Adjusted for less random, more focused output
    top_p=0.9,
    max_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH, # Ensure it doesn't exceed model capacity
    # The following are common vLLM SamplingParams, add if needed by fast_generate
    # n=1, # Number of output sequences to return
    # presence_penalty=0.0,
    # frequency_penalty=0.0,
    # use_beam_search=False,
)

print(f"\nGenerating response for test prompt:\n{test_prompt_text}")
if hasattr(model, "fast_generate"):
    # Unsloth's fast_generate might expect a LoraRequest if LoRAs are managed that way
    # For a model already PEFT-tuned, it might use the existing adapters.
    # Check Unsloth docs for `fast_generate` with active LoRA layers.
    # lora_request = model.load_lora(final_lora_path) # If needed to specify LoRA for fast_generate
    
    # Assuming LoRA is active on the model:
    outputs = model.fast_generate(
        test_prompt_text,
        sampling_params=sampling_params,
        # lora_request=lora_request # if lora_request is needed
    )
    if outputs and outputs[0].outputs:
         generated_text = outputs[0].outputs[0].text
         print("\nGenerated Text (using fast_generate):")
         print(generated_text)
    else:
        print("Fast generation did not produce expected output structure.")
        generated_text = "Error in generation."

else:
    print("`fast_generate` not available. Using Hugging Face `generate` method.")
    # Fallback to standard Hugging Face generation if fast_generate is not present or fails
    # Ensure model is on the correct device
    device = next(model.parameters()).device
    inputs = tokenizer(test_prompt_text, return_tensors="pt").to(device)
    
    # Note: The model object from GRPOTrainer might be the PeftModel.
    # If it's a GRPOTrainer.model.model, it's the base model.
    # Ensure you're calling generate on the model with adapters.
    # The `model` variable should be the PeftModel.
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\nGenerated Text (using HF generate):")
    print(generated_text)


# --- Saving the Full Model (Merged) ---
# Unsloth provides methods to save merged models in different precisions.
# This merges LoRA weights into the base model. Requires more memory.
# Consider doing this on a machine with more RAM if current one is constrained.

# Save to 16-bit precision (common for sharing)
merged_model_path_16bit = f"./{NEW_MODEL_NAME_BASE}_merged_16bit"
print(f"\nSaving merged model to 16-bit at: {merged_model_path_16bit}")
try:
    # Ensure model is on CPU if merging on a memory-constrained GPU setup
    # model.cpu()
    # tokenizer.save_pretrained(merged_model_path_16bit)
    # model.save_pretrained_merged(merged_model_path_16bit, tokenizer, save_method="merged_16bit")
    
    # Simpler way if model is already configured by Unsloth:
    model.save_pretrained_merged(merged_model_path_16bit, tokenizer, save_method="merged_16bit")
    print(f"Model saved to {merged_model_path_16bit}")
except Exception as e:
    print(f"Failed to save merged 16-bit model: {e}")
    print("You might need to ensure the model is on CPU or have enough GPU RAM for merging.")

# Save to 4-bit precision (for inference with Unsloth)
# merged_model_path_4bit = f"./{NEW_MODEL_NAME_BASE}_merged_4bit"
# print(f"\nSaving merged model to 4-bit at: {merged_model_path_4bit}")
# try:
#     model.save_pretrained_merged(merged_model_path_4bit, tokenizer, save_method="merged_4bit_forced") # or "merged_4bit"
#     print(f"Model saved to {merged_model_path_4bit}")
# except Exception as e:
#     print(f"Failed to save merged 4-bit model: {e}")


# --- Pushing to Hugging Face Hub ---
# Replace "YOUR_HF_USERNAME" and "YOUR_HF_TOKEN" with your actual credentials.
# You need to be logged in via `huggingface-cli login` or provide a token.

# HF_USERNAME = "YOUR_HF_USERNAME"  # Replace!
# HF_TOKEN = "YOUR_HF_TOKEN"      # Replace! Create a token with write access on HF.
# HF_MODEL_ID_16BIT = f"{HF_USERNAME}/{NEW_MODEL_NAME_BASE}-16bit"
# HF_MODEL_ID_GGUF = f"{HF_USERNAME}/{NEW_MODEL_NAME_BASE}-GGUF"

# Push merged 16-bit model
# print(f"\nAttempting to push 16-bit merged model to Hugging Face Hub: {HF_MODEL_ID_16BIT}")
# try:
#     # Ensure you have the merged model loaded if you cleared it or are in a new session
#     # model.push_to_hub_merged(HF_MODEL_ID_16BIT, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
#     print(f"Successfully pushed 16-bit model to {HF_MODEL_ID_16BIT}")
# except Exception as e:
#     print(f"Failed to push 16-bit model to Hub: {e}")
#     print("Ensure you have replaced placeholders, have correct permissions, and are logged in.")

# Push GGUF quantized model (useful for llama.cpp)
# print(f"\nAttempting to push GGUF model to Hugging Face Hub: {HF_MODEL_ID_GGUF}")
# try:
#     model.push_to_hub_gguf(
#         HF_MODEL_ID_GGUF,
#         tokenizer,
#         quantization_method=["q4_k_m", "q8_0", "q5_k_m"], # Example quant methods
#         token=HF_TOKEN
#     )
#     print(f"Successfully pushed GGUF model to {HF_MODEL_ID_GGUF}")
# except Exception as e:
#     print(f"Failed to push GGUF model to Hub: {e}")
#     print("Ensure you have replaced placeholders and have necessary tools for GGUF conversion if not built-in.")

print("\n--- Script Finished ---")

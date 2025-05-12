# unsloth_grpo_gsm8k.py
# Standalone script for GRPO fine-tuning with Unsloth based on the provided example.

# --- Installation (run these in your terminal if not already installed) ---
# pip install "unsloth[colab-new]" # For Google Colab
# pip install "unsloth[conda-new]" # For Conda
# pip install "unsloth[pytorch-new]" # For PyTorch
# pip install vllm pillow
# pip install datasets trl transformers peft accelerate bitsandbytes

import torch
import re
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams # For fast_generate, if used directly with vLLM engine
import wandb

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" # Unsloth example model
DATASET_ID = "openai/gsm8k"
DATASET_SPLIT = "main" # GSM8K uses 'main' for configuration and 'train'/'test' for splits
NEW_MODEL_NAME_BASE = "Gemma3-1B-GRPO-Unsloth-GSM8K" # Base name for saving
WANDB_PROJECT = "Unsloth-GRPO-GSM8K"

# Unsloth & LoRA Configuration
MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
LORA_RANK = 32         # Larger rank = smarter, but slower. Suggested 8, 16, 32, 64, 128
LOAD_IN_4BIT = True    # Use 4-bit quantization
TARGET_MODULES = [     # Modules to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Weights & Biases Login (Optional) ---
try:
    wandb.login()
    WANDB_ENABLED = True
    print("Successfully logged into Weights & Biases.")
except Exception as e:
    print(f"Could not log in to W&B: {e}")
    print("Proceeding without W&B logging. Set WANDB_ENABLED=False or fix login to enable.")
    WANDB_ENABLED = False

# --- Load Model and Tokenizer with Unsloth ---
print(f"Loading model: {MODEL_NAME} with Unsloth")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detects, or torch.bfloat16 for Ampere+
    load_in_4bit=LOAD_IN_4BIT,
    # token = "hf_...", # use token if using private models
    fast_inference=True,  # Enable vLLM fast inference with model.fast_generate
    # max_lora_rank=LORA_RANK, # Not a direct param for from_pretrained, handled by get_peft_model
    # gpu_memory_utilization=0.6, # Can be useful, but let's start without it
)
print("Model and tokenizer loaded.")

print("Applying LoRA configuration...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_RANK, # Often set to r or 2*r
    lora_dropout=0, # Optional
    bias="none",    # Optional
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for Unsloth's implementation
    random_state=3407,
    # max_seq_length=MAX_SEQ_LENGTH, # Handled by from_pretrained
)
print("LoRA configured.")
model.print_trainable_parameters()

# Set pad token if not already set
if tokenizer.pad_token is None:
    print("Pad token not set. Setting to EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    # Note: Unsloth model config might handle pad_token_id internally,
    # but explicit setting for tokenizer is good practice.
    # model.config.pad_token_id = tokenizer.eos_token_id # Usually handled by FastLanguageModel

# --- Data Preparation ---
print(f"Preparing dataset: {DATASET_ID}")

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML_COT_FORMAT is not directly used in dataset prep here, but good to have for generation
# XML_COT_FORMAT = """\
# <reasoning>
# {reasoning}
# </reasoning>
# <answer>
# {answer}
# </answer>
# """

def extract_xml_answer(text: str) -> str:
    parts = text.split("<answer>")
    if len(parts) > 1:
        answer = parts[-1].split("</answer>")[0]
        return answer.strip()
    return "" # Return empty if format not found

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train", tokenizer_for_chat_template=None) -> Dataset:
    dataset = load_dataset(DATASET_ID, DATASET_SPLIT, split=split) # 'main' config, 'train' split
    
    def format_example(example):
        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        # The prompt for the model will be the formatted chat history
        # The `answer` field will be used by the correctness reward function
        return {
            "prompt": tokenizer_for_chat_template.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": extract_hash_answer(example["answer"]), # Ground truth answer
        }

    # Filter out examples where answer extraction fails
    dataset = dataset.filter(lambda x: extract_hash_answer(x["answer"]) is not None)
    
    # Apply formatting
    # We need the tokenizer here to apply the chat template for the 'prompt'
    if tokenizer_for_chat_template is None:
        raise ValueError("Tokenizer must be provided to get_gsm8k_questions for chat templating.")
        
    dataset = dataset.map(lambda x: format_example(x))
    
    # GRPO Trainer expects 'prompt' (string) and 'answer' (string, for correctness_reward_func)
    # It also expects 'chosen' and 'rejected' for standard DPO/IPO, but GRPO generates these.
    # For GRPO, the 'prompt' will be used to generate completions.
    # The 'answer' column here is the ground truth, used by `correctness_reward_func`.
    return dataset

train_dataset = get_gsm8k_questions(split="train", tokenizer_for_chat_template=tokenizer)
# test_dataset = get_gsm8k_questions(split="test", tokenizer_for_chat_template=tokenizer) # Optional for eval

print("Dataset prepared:")
print(train_dataset)
if len(train_dataset) > 0:
    print("\nSample entry:")
    print(f"Prompt: {train_dataset[0]['prompt']}")
    print(f"Answer: {train_dataset[0]['answer']}")


# --- Reward Functions ---
print("Defining reward functions...")

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # 'completions' is a list of lists of dicts: [[{'role': 'assistant', 'content': '...'}], ...]
    # 'answer' is a list of ground truth answers passed from the dataset
    responses = [comp[0]["content"] for comp in completions] # Assuming single turn assistant response
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Debug print for one example per batch
    if prompts and responses and answer and extracted_responses:
        print("-" * 20)
        # prompt_text = prompts[0] # The prompt is already formatted string
        # print(f"Prompt (for reward): \n{prompt_text}") # This is the input prompt to the model
        print(f"GT Answer (for reward): {answer[0]}")
        print(f"Generated Completion (for reward): \n{responses[0]}")
        print(f"Extracted Model Answer (for reward): {extracted_responses[0]}")
        print("-" * 20)
        
    return [2.0 if resp == gt_ans else 0.0 for resp, gt_ans in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$" # Allow optional trailing newline
    responses = [comp[0]["content"] for comp in completions]
    matches = [bool(re.match(pattern, r, re.DOTALL)) for r in responses] # re.DOTALL for multiline reasoning
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [comp[0]["content"] for comp in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses] # Use search for flexibility
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    # More robust counting, allowing for variations in newlines around tags
    if re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL):
        count += 0.25 # Presence of reasoning block
    if re.search(r"<answer>.*?</answer>", text, re.DOTALL):
        count += 0.25 # Presence of answer block

    # Penalize content outside the final answer tag
    # This is a bit tricky; the example logic was:
    # count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    # count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    # Let's simplify: penalize if there's significant text after the last </answer>
    match_answer_end = list(re.finditer(r"</answer>", text, re.DOTALL))
    if match_answer_end:
        last_match_end = match_answer_end[-1].end()
        trailing_text = text[last_match_end:].strip()
        if trailing_text:
            count -= 0.1 # Simple penalty for any trailing text
            count -= len(trailing_text) * 0.001 # Scale penalty by length
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [comp[0]["content"] for comp in completions]
    return [count_xml(c) for c in contents]

# --- Training Arguments ---
print("Defining training arguments...")
MAX_PROMPT_LENGTH = 512 # Max length for the prompt part
# Max completion length will be MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
# This ensures the total length does not exceed MAX_SEQ_LENGTH

training_args = GRPOConfig(
    output_dir=f"{NEW_MODEL_NAME_BASE}-checkpoints",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", # Unsloth recommended optimizer
    logging_steps=10, # Log more frequently
    per_device_train_batch_size=1, # As in example
    gradient_accumulation_steps=4,  # Increase for smoother training, e.g., 4
    num_generations=6,  # Number of completions to generate per prompt
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
    # num_train_epochs=1, # Set for a full training run if max_steps is not set
    max_steps=250,  # For a quicker run, as in example. Comment out for full epochs.
    save_steps=250, # Save checkpoints
    max_grad_norm=0.1, # Gradient clipping
    report_to=["wandb"] if WANDB_ENABLED else [],
    remove_unused_columns=False, # Important for custom datasets
    bf16=not LOAD_IN_4BIT, # Use bf16 if not using 4-bit, requires Ampere+
    fp16=False, # Don't use fp16 if using bf16 or 4bit
    gradient_checkpointing=True, # Already set in get_peft_model via use_gradient_checkpointing
    # use_vllm=True, # This was a GRPOConfig param in some TRL versions, now auto-detected or relies on Unsloth's fast_generate
    reward_adapter_name="default", # For multi-adapter reward modeling if ever needed
    save_total_limit=2, # Limit number of checkpoints
)
print("Training arguments defined.")

# --- Initialize Trainer ---
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model, # Unsloth model
    tokenizer=tokenizer, # Unsloth tokenizer
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset, # Optional
)
print("GRPOTrainer initialized.")

# --- Start Training ---
print("Starting training...")
if WANDB_ENABLED:
    try:
        wandb.init(project=WANDB_PROJECT, config=training_args)
        print(f"W&B run initialized for project: {WANDB_PROJECT}")
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}")
        print("Continuing training without W&B.")
        trainer.args.report_to = []

trainer.train()
print("Training finished.")

# --- Save LoRA Adapters (Unsloth specific) ---
# This saves only the LoRA weights, not the merged model.
# Useful if you want to load the base model and apply LoRA later.
lora_save_path = f"{NEW_MODEL_NAME_BASE}-lora"
print(f"Saving LoRA adapters to {lora_save_path}...")
model.save_lora(lora_save_path) # Unsloth's method
print("LoRA adapters saved.")


# --- Testing the Model (after training) ---
print("\n--- Text Generation Example (after training) ---")
# For testing, it's often best to merge LoRA weights into the base model.
# Unsloth's fast_generate can sometimes use LoRA on the fly with lora_request,
# but for general use and saving, merging is common.

# Option 1: Use the model directly if it's already the merged one (some trainers merge at end)
# Option 2: Explicitly merge if needed (Unsloth handles this with save_pretrained_merged)
# Option 3: Use fast_generate with lora_request if supported and adapters not merged
# For this example, we will demonstrate generation after merging (see save section).

# Let's prepare a prompt for testing:
test_prompt_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
]
formatted_test_prompt = tokenizer.apply_chat_template(
    test_prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(f"\nTest Prompt:\n{formatted_test_prompt}")

# For generation, we typically want the merged model.
# If you want to test with the current LoRA-adapted model (pre-merge):
# Note: Ensure `fast_inference=True` was set for `FastLanguageModel.from_pretrained`
# And LoRA adapters are active.
# model.eval() # Set to evaluation mode

# Using Unsloth's fast_generate with the trained LoRA model (before explicit merging for save)
# This requires vLLM backend to be active.
try:
    print("\nGenerating text using fast_generate (with trained LoRA)...")
    # FastLanguageModel might need lora_request if adapters are not automatically active
    # However, after training, the PEFT model should have adapters active.

    # Define sampling parameters for Unsloth's fast_generate
    # These are different from transformers' generate kwargs
    # and are vLLM's SamplingParams
    unsloth_sampling_params = SamplingParams(
        temperature=0.7, # Example, adjust as needed
        top_p=0.9,       # Example
        max_tokens=MAX_SEQ_LENGTH - len(tokenizer.encode(formatted_test_prompt)), # Max new tokens
        # stop=["</answer>"] # Optional: stop generation after this token
    )
    
    # fast_generate expects a list of prompts if batching, or a single string
    outputs = model.fast_generate(
        formatted_test_prompt,
        sampling_params=unsloth_sampling_params,
        # lora_request=LoRARequest("grpo_lora", 1, lora_save_path) # If loading specific LoRA
    )
    
    generated_text = outputs[0].outputs[0].text # Accessing the text from vLLM output structure
    print("\nGenerated Text (fast_generate):")
    print(generated_text)

except Exception as e:
    print(f"Could not perform text generation with fast_generate: {e}")
    print("This might be due to vLLM setup or model state. Trying with Hugging Face pipeline as fallback after merge.")


# --- Save Full Model (Merged) ---
# Unsloth provides convenient methods to save the merged model.
# This is generally what you'd want for deployment or sharing.

# Option 1: Save to 16-bit precision (recommended for quality)
merged_model_path_16bit = f"{NEW_MODEL_NAME_BASE}-merged-16bit"
print(f"\nSaving merged model to 16-bit at {merged_model_path_16bit}...")
try:
    # The `model` object from GRPOTrainer is the PeftModel.
    # Unsloth's `save_pretrained_merged` expects the FastLanguageModel instance
    # that has been adapted with PEFT.
    # If `trainer.model` is the PeftModel, ensure it's the one with `save_pretrained_merged`.
    # FastLanguageModel itself has these methods.
    
    # If trainer.model is a PeftModel, it should have an `model.model` to get the underlying Unsloth model
    # or Unsloth's PeftModel wrapper might directly have this.
    # Let's assume `model` (the one we applied get_peft_model to) is the correct object.
    model.save_pretrained_merged(merged_model_path_16bit, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved to {merged_model_path_16bit}")

    # Test generation with the saved and reloaded merged model using Transformers pipeline
    print(f"\nLoading merged 16-bit model from {merged_model_path_16bit} for pipeline test...")
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model=merged_model_path_16bit, device_map="auto", torch_dtype=torch.bfloat16)
    
    pipeline_output = pipe(formatted_test_prompt, max_new_tokens=MAX_SEQ_LENGTH - len(tokenizer.encode(formatted_test_prompt)), do_sample=True, temperature=0.7, top_p=0.9)
    print("\nGenerated Text (pipeline on merged 16-bit model):")
    print(pipeline_output[0]['generated_text'])

except Exception as e:
    print(f"Error during 16-bit model saving or pipeline test: {e}")

# Option 2: Save to 4-bit precision (for smaller size, if `load_in_4bit=True` was used for training)
if LOAD_IN_4BIT:
    merged_model_path_4bit = f"{NEW_MODEL_NAME_BASE}-merged-4bit"
    print(f"\nSaving merged model to 4-bit at {merged_model_path_4bit}...")
    try:
        model.save_pretrained_merged(merged_model_path_4bit, tokenizer, save_method="merged_4bit_forced") # Or "merged_4bit"
        print(f"Merged 4-bit model saved to {merged_model_path_4bit}")
    except Exception as e:
        print(f"Error during 4-bit model saving: {e}")
else:
    print("\nSkipping 4-bit save as LOAD_IN_4BIT was False.")


# --- Push to Hugging Face Hub (Optional) ---
# Ensure you are logged in: `huggingface-cli login`
# Replace "your-hf-username" with your actual Hugging Face username.
HF_USERNAME = "your-hf-username" # <<<<----- REPLACE THIS
HF_MODEL_NAME_16BIT = f"{HF_USERNAME}/{NEW_MODEL_NAME_BASE}-merged-16bit"
HF_MODEL_NAME_4BIT = f"{HF_USERNAME}/{NEW_MODEL_NAME_BASE}-merged-4bit"
HF_MODEL_NAME_GGUF = f"{HF_USERNAME}/{NEW_MODEL_NAME_BASE}-GGUF"

PUSH_TO_HUB = False # Set to True to enable pushing

if PUSH_TO_HUB:
    hf_token = input("Enter your Hugging Face Hub token (or press Enter if already configured globally): ")
    if not hf_token: hf_token = None # Use global token if empty

    # Push 16-bit model
    print(f"\nAttempting to push 16-bit model to {HF_MODEL_NAME_16BIT}...")
    try:
        model.push_to_hub_merged(HF_MODEL_NAME_16BIT, tokenizer, save_method="merged_16bit", token=hf_token)
        print("16-bit model pushed to Hub.")
    except Exception as e:
        print(f"Failed to push 16-bit model to Hub: {e}")

    # Push 4-bit model (if saved and applicable)
    if LOAD_IN_4BIT:
        print(f"\nAttempting to push 4-bit model to {HF_MODEL_NAME_4BIT}...")
        try:
            # We need to load the 4-bit saved model first if we want to push that specific one,
            # or specify the quantization method if push_to_hub_merged supports it directly.
            # The example uses `save_method="merged_4bit_forced"` for saving,
            # so `push_to_hub_merged` should ideally support a similar parameter or push from the saved path.
            # For simplicity, let's assume it pushes the current state quantized to 4-bit if possible
            # or that we'd load the saved 4-bit model then push.
            # The documentation suggests push_to_hub_merged can take save_method.
            model.push_to_hub_merged(HF_MODEL_NAME_4BIT, tokenizer, save_method="merged_4bit_forced", token=hf_token)
            print("4-bit model pushed to Hub.")
        except Exception as e:
            print(f"Failed to push 4-bit model to Hub: {e}")
    
    # Push GGUF formats
    print(f"\nAttempting to push GGUF model to {HF_MODEL_NAME_GGUF}...")
    try:
        # Common quantization methods for GGUF.
        # You might need to install llama-cpp-python with specific options for GGUF creation to work.
        # pip install llama-cpp-python
        # Ensure the model is in a compatible state (e.g., not 4-bit quantized in a way GGUF doesn't like)
        # Sometimes it's better to push from a 16-bit merged model.
        # For now, assuming `model` (the LoRA adapted one) can be source for GGUF.
        model.push_to_hub_gguf(
            HF_MODEL_NAME_GGUF, 
            tokenizer, 
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"], # Example methods
            token=hf_token
        )
        print("GGUF models pushed to Hub.")
    except Exception as e:
        print(f"Failed to push GGUF model to Hub: {e}")
else:
    print("\nSkipping Hugging Face Hub push as PUSH_TO_HUB is False.")

print("\n--- Script Finished ---")

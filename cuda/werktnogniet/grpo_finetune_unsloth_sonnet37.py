#!/usr/bin/env python3
import os
import re
from unsloth import FastLanguageModel
import torch
import argparse
from datasets import load_dataset, Dataset
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
import wandb

# --- Configuration ---
MODEL_ID = "google/gemma-3-1b-it"
DATASET_ID = "openai/gsm8k"
DATASET_SPLIT = "main"
NEW_MODEL_NAME = "gemma-3-1b-grpo-reasoning"
WANDB_PROJECT = "GRPO-Reasoning"
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 256
LORA_RANK = 32

# --- System Prompts and Templates ---
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# --- Helper Functions ---
def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    print(f"Loading dataset: {DATASET_ID}/{DATASET_SPLIT}")
    data = load_dataset(DATASET_ID, DATASET_SPLIT)[split]
    print("Dataset loaded, preparing prompts...")
    
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    print(f"Dataset prepared with {len(data)} examples")
    return data

# --- Reward Functions ---
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Print sample for debugging
    if kwargs.get("verbose", False):
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>") == 1:
        count += 0.125
    if text.count("</reasoning>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
        count -= len(text.split("</answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def main():
    parser = argparse.ArgumentParser(description="Train a model with GRPO using Unsloth")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID to fine-tune")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save model")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--max_steps", type=int, default=250, help="Maximum training steps")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_id", type=str, default=None, help="Hugging Face Hub ID for pushing")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()

    # --- Weights & Biases Setup ---
    if args.wandb:
        try:
            wandb.login()
            print(f"Weights & Biases logging enabled for project: {WANDB_PROJECT}")
            wandb_enabled = True
        except Exception as e:
            print(f"Could not log in to W&B: {e}")
            print("Proceeding without W&B logging.")
            wandb_enabled = False
    else:
        wandb_enabled = False

    # --- Load Model with Unsloth ---
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
    )
    print("Model loaded successfully.")

    # --- Configure LoRA ---
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA configured.")

    # --- Prepare Dataset ---
    dataset = get_gsm8k_questions()

    # --- Training Arguments ---
    print("Defining training arguments...")
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        max_steps=args.max_steps,
        save_steps=args.max_steps,
        max_grad_norm=0.1,
        report_to="wandb" if wandb_enabled else "none",
        output_dir=args.output_dir,
    )
    print("Training arguments defined.")

    # --- Initialize Trainer ---
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    print("GRPOTrainer initialized.")

    # --- Start Training ---
    print("Starting training...")
    if wandb_enabled:
        try:
            wandb.init(project=WANDB_PROJECT)
            print(f"W&B run initialized for project: {WANDB_PROJECT}")
        except Exception as e:
            print(f"Failed to initialize W&B run: {e}")
            print("Continuing training without W&B.")
            trainer.args.report_to = "none"

    trainer.train()
    print("Training finished.")

    # --- Save Model ---
    print("Saving LoRA adapters...")
    model.save_lora(os.path.join(args.output_dir, "grpo_saved_lora"))
    print("LoRA adapters saved.")

    # --- Save Merged Model --- 
    print("Saving merged model...")
    model.save_pretrained_merged(os.path.join(args.output_dir, "model"), tokenizer, save_method="merged_16bit")
    print("Merged model saved.")

    # --- Push to Hub ---
    if args.push_to_hub and args.hub_id and args.token:
        try:
            print(f"Pushing model to Hub: {args.hub_id}")
            model.push_to_hub_merged(
                args.hub_id, 
                tokenizer, 
                save_method="merged_16bit", 
                token=args.token
            )
            print("Model pushed to Hub successfully.")
        except Exception as e:
            print(f"Failed to push model to Hub: {e}")
    
    # --- Test Model ---
    print("\n--- Testing Model ---")
    try:
        test_question = "If a train travels at 120 km/h for 2.5 hours, how far does it go?"
        
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": test_question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
        
        lora_path = os.path.join(args.output_dir, "grpo_saved_lora")
        output = (
            model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=model.load_lora(lora_path),
            )[0]
            .outputs[0]
            .text
        )

        print("\nGenerated Text:")
        print(output)
        print("\nExtracted Answer:")
        print(extract_xml_answer(output))

    except Exception as gen_e:
        print(f"Could not perform text generation: {gen_e}")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
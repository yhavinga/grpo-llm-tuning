import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments
from trl import GRPOConfig, GRPOTrainer

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
DATASET_ID = "mlabonne/smoltldr"
NEW_MODEL_NAME = "SmolGRPO-135M"
WANDB_PROJECT = "GRPO"
IDEAL_COMPLETION_LENGTH = 50

# --- Weights & Biases Login ---
try:
    wandb.login()
    WANDB_ENABLED = True
except Exception as e:
    print(f"Could not log in to W&B: {e}")
    print("Proceeding without W&B logging.")
    WANDB_ENABLED = False

# --- Load Dataset ---
print(f"Loading dataset: {DATASET_ID}")
dataset = load_dataset(DATASET_ID)
print("Dataset loaded:")
print(dataset)

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # Use bfloat16 as specified in original training args
    device_map="auto",
    attn_implementation="flash_attention_2",
)
print("Model loaded.")

print(f"Loading tokenizer for model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer loaded.")

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    print("Pad token set to EOS token.")

# --- LoRA Configuration ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print("LoRA configured.")
print(model.print_trainable_parameters())


# --- Reward Function Definition ---
def reward_len(completions, **kwargs):
    """Simple reward function favoring completions near the ideal length."""
    return [-abs(IDEAL_COMPLETION_LENGTH - len(completion)) for completion in completions]

print(f"Reward function defined targeting length: {IDEAL_COMPLETION_LENGTH}")

# --- Training Arguments ---
print("Defining training arguments...")
training_args = GRPOConfig(
    output_dir=NEW_MODEL_NAME, # Directory to save checkpoints and final model
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True, # Use bfloat16 mixed precision
    logging_steps=1,
    report_to=["wandb"] if WANDB_ENABLED else [],
    remove_unused_columns=False,
    # push_to_hub=True, # Uncomment to push during training
    # hub_model_id=f"YourHuggingFaceUsername/{NEW_MODEL_NAME}", # Replace with your HF username
)
print("Training arguments defined.")

# --- Initialize Trainer ---
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)
print("GRPOTrainer initialized.")

# --- Start Training ---
print("Starting training...")
if WANDB_ENABLED:
    try:
        wandb.init(project=WANDB_PROJECT)
        print(f"W&B run initialized for project: {WANDB_PROJECT}")
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}")
        print("Continuing training without W&B.")
        trainer.args.report_to = [] # Disable reporting if init fails

trainer.train()
print("Training finished.")

# --- Save and Publish Model ---
print("Merging LoRA adapters and unloading...")
# Ensure the model is on CPU before merging if device_map='auto' was used
# and you might run out of GPU memory during merge.
# Merging requires substantial memory. Consider doing this on a machine with more RAM/GPU RAM.
# trainer.model.cpu() # Optional: Move model to CPU if needed
merged_model = trainer.model.merge_and_unload()
print("Model merged.")

try:
    print(f"Pushing model to Hub: {NEW_MODEL_NAME}")
    # You might need to log in to Hugging Face Hub via `huggingface-cli login` first
    merged_model.push_to_hub(
        NEW_MODEL_NAME, private=False, tags=["GRPO", "Reasoning-Course"]
    )
    tokenizer.push_to_hub(
        NEW_MODEL_NAME, private=False, tags=["GRPO", "Reasoning-Course"]
    )
    print("Model and tokenizer pushed to Hub successfully.")
except Exception as e:
    print(f"Failed to push model to Hub: {e}")
    print(f"You can manually save the model using: merged_model.save_pretrained('{NEW_MODEL_NAME}')")
    print(f"And the tokenizer using: tokenizer.save_pretrained('{NEW_MODEL_NAME}')")
    # Fallback to local save
    try:
        print(f"Saving model locally to ./{NEW_MODEL_NAME}")
        merged_model.save_pretrained(NEW_MODEL_NAME)
        tokenizer.save_pretrained(NEW_MODEL_NAME)
        print("Model and tokenizer saved locally.")
    except Exception as save_e:
        print(f"Failed to save model locally: {save_e}")


# --- Generate Text ---
print("\n--- Text Generation Example ---")
# Use the merged model directly
# If push_to_hub failed or you want to use the local version:
# generator_model = AutoModelForCausalLM.from_pretrained(NEW_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
# generator_tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_NAME)
# generator = pipeline("text-generation", model=generator_model, tokenizer=generator_tokenizer)

# Or load from Hub if pushed successfully
# Make sure the model name matches where it was pushed
# hub_model_id = f"YourHuggingFaceUsername/{NEW_MODEL_NAME}" # Or just NEW_MODEL_NAME if pushed to your default namespace
try:
    print(f"Loading pipeline for generation with model: {NEW_MODEL_NAME}")
    generator = pipeline("text-generation", model=NEW_MODEL_NAME) # Assumes push_to_hub was successful
    print("Pipeline loaded.")

    prompt = """
# A long document about the Cat

The cat (Felis catus), also referred to as the domestic cat or house cat, is a small
domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
Advances in archaeology and genetics have shown that the domestication of the cat occurred
in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
freely as a feral cat avoiding human contact. It is valued by humans for companionship and
its ability to kill vermin. Its retractable claws are adapted to killing small prey species
such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
and its night vision and sense of smell are well developed. It is a social species,
but a solitary hunter and a crepuscular predator. Cat communication includes
vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
well as body language. It can hear sounds too faint or too high in frequency for human ears,
such as those made by small mammals. It secretes and perceives pheromones.
"""

    messages = [
        {"role": "user", "content": prompt},
    ]

    generate_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.5,
        "min_p": 0.1,
    }

    print("\nGenerating text...")
    generated_text = generator(messages, **generate_kwargs) # Pass kwargs correctly

    print("\nGenerated Text:")
    # The output format might vary slightly depending on the pipeline version
    if isinstance(generated_text, list) and len(generated_text) > 0 and isinstance(generated_text[0], dict):
         print(generated_text[0].get('generated_text', 'No generated text found in expected format.'))
    else:
        print(generated_text)

except Exception as gen_e:
    print(f"Could not perform text generation: {gen_e}")
    print("This might happen if the model was not pushed to the Hub or saved locally correctly.")

print("\n--- Script Finished ---")

import sys
import os
import json
from datasets import load_dataset

# Ensure the 'backend' directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.model_loader import load_base_model
from app.services.lora import integrate_lora
from app.services.train import train_model

print("Starting the training script...")

# Step 1: Load the Base Model and Tokenizer
print("Step 1: Load the Base Model and Tokenizer...")
model_name = "EleutherAI/gpt-neo-125M"
model, tokenizer = load_base_model(model_name)
print(f"Step 1 Complete: Successfully loaded base model '{model_name}'.")

# Set a padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding if PAD token is not defined
    print("Added a pad token to the tokenizer.")

# Step 2: Load LoRA Configuration
print("Step 2: Load LoRA Configuration...")
lora_config_path = os.path.join(os.path.dirname(__file__), "../config/lora_config.json")
try:
    with open(lora_config_path, "r") as config_file:
        lora_params = json.load(config_file)
    print(f"Step 2 Complete: Loaded LoRA configuration from '{lora_config_path}'.")
except FileNotFoundError:
    print(f"Step 2 Error: Configuration file '{lora_config_path}' not found.")
    sys.exit(1)

# Step 3: Integrate LoRA into the Model
print("Step 3: Integrate LoRA into the Model...")
lora_model = integrate_lora(model, lora_params)
print("Step 3 Complete: LoRA successfully integrated into the model.")

# Step 4: Prepare Dataset
print("Step 4: Load and Prepare Dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print("Step 4.1: Dataset loaded. Starting tokenization...")

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Set labels to be equal to input_ids for language modeling tasks
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("Step 4 Complete: Dataset tokenization complete.")

# Step 5: Train the Model
print("Step 5: Start Model Training...")
output_dir = os.path.join(os.path.dirname(__file__), "../output_dir")

try:
    train_model(lora_model, tokenizer, tokenized_dataset, output_dir)
    print(f"Step 5 Complete: Model training complete. Checkpoints saved in '{output_dir}'.")
except Exception as e:
    print(f"Step 5 Error: An error occurred during training: {e}")
    sys.exit(1)

print("Training script completed successfully.")

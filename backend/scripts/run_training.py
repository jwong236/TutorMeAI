from datasets import load_dataset
from app.services.train import train_model
from app.services.model_loader import load_base_model
from app.services.lora import integrate_lora

"""
This script fine-tunes a GPT-Neo model with LoRA integration on the Wikitext-2 dataset.

Steps:
1. Load the Wikitext-2 dataset, GPT-Neo base model, and tokenizer.
2. Tokenize the dataset and ensure labels are prepared for training.
3. Integrate LoRA into the model for efficient parameter updates.
4. Fine-tune the model and save the trained weights.

Outputs:
- Trained model weights saved to the specified output directory.
"""


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Load base model and tokenizer
    model_name = "EleutherAI/gpt-neo-125M"
    print(f"Loading base model '{model_name}'...")
    model, tokenizer = load_base_model(model_name)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Added pad token to tokenizer.")

    # Tokenize dataset
    print("Tokenizing dataset...")

    # concern: the dataset has the keys text, input_ids, and attention_mask, but transformers expects to have a labels key
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True
    )  # Keys are text, input_ids, attention_mask

    # Integrate LoRA into the model
    print("Integrating LoRA into the model...")
    lora_params = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": [
            "attn.attention.k_proj",
            "attn.attention.q_proj",
            "attn.attention.v_proj",
        ],
        "lora_dropout": 0.1,
        "bias": "none",
    }
    model = integrate_lora(model, lora_params)

    # Train the model
    print("Starting training...")
    output_dir = "./results"
    trainer = train_model(model, tokenizer, tokenized_dataset, output_dir)

    print("Training completed. Model saved to ./results.")


if __name__ == "__main__":
    main()

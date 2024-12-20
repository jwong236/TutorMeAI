from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset


def train_model(model, tokenizer, dataset, output_dir):
    """
    Trains the model using the given dataset and saves the results.

    Args:
        model: The language model (with LoRA).
        tokenizer: The tokenizer corresponding to the model.
        dataset: The dataset used for training (dictionary with fields 'text', 'input_ids', 'attention_mask').
        output_dir (str): The directory where trained model checkpoints are saved.
    """

    # Split the dataset into training and validation sets
    split_data = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_data["train"]
    val_dataset = split_data["test"]

    # Define a data collator to handle padding dynamically during batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=3,
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Return trainer for further inspection or evaluation
    return trainer

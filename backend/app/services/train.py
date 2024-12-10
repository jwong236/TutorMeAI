from transformers import TrainingArguments, Trainer

def train_model(model, tokenizer, dataset, output_dir):
    """
    Trains the model using the given dataset and saves the results.

    Args:
        model: The language model (with LoRA).
        tokenizer: The tokenizer corresponding to the model.
        dataset: The dataset used for training.
        output_dir (str): The directory where trained model checkpoints are saved.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()

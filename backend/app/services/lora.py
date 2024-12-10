from peft import LoraConfig, get_peft_model

def integrate_lora(model, lora_params):
    """
    Integrates a LoRA adapter into the base model.
    
    Args:
        model: The base language model.
        lora_params (dict): Dictionary containing LoRA configuration parameters.

    Returns:
        model: The model with the integrated LoRA adapter.
    """
    config = LoraConfig(
        r=lora_params.get('r', 8),
        lora_alpha=lora_params.get('lora_alpha', 16),
        target_modules=lora_params.get('target_modules', ["query", "value"]),
        lora_dropout=lora_params.get('lora_dropout', 0.1),
        bias=lora_params.get('bias', "none"),
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

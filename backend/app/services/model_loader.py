from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(model_name):
    """
    Loads the base model and tokenizer.
    
    Args:
        model_name (str): Name of the model to load

    Returns:
        model: The pre-trained language model.
        tokenizer: The corresponding tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

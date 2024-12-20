from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model_name = "EleutherAI/gpt-neo-125M"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load LoRA model
lora_checkpoint = "./output_dir/checkpoint-500"
model_with_lora = PeftModel.from_pretrained(base_model, lora_checkpoint)

# Compare outputs
inputs = tokenizer(
    "The history of artificial intelligence began in the mid-20th century.",
    return_tensors="pt",
)

# Base model output
print("Base Model Output:")
base_outputs = base_model.generate(**inputs, max_length=50)
print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

# LoRA model output
print("LoRA Model Output:")
lora_outputs = model_with_lora.generate(**inputs, max_length=50)
print(tokenizer.decode(lora_outputs[0], skip_special_tokens=True))

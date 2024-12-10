from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output_dir/checkpoint-1000")
tokenizer = AutoTokenizer.from_pretrained("./output_dir/checkpoint-1000")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
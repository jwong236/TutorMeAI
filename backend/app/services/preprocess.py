def filter_invalid_samples(example):
    if isinstance(example["text"], list):
        example["text"] = " ".join(example["text"])
    return isinstance(example["text"], str) and len(example["text"].strip()) > 0

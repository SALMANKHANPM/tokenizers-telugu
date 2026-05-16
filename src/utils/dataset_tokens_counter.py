from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")

dataset = load_dataset("nvidia/OpenCodeInstruct", split="train")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)

total = 0
max = 0

for sample in tqdm(dataset):
    text = sample['input'] + sample['output']
    tokens = tokenizer.encode(text)
    total += len(tokens)
    if len(tokens) > max:
        max = len(tokens)

print("Total tokens:", total)
print("Max tokens:", max)


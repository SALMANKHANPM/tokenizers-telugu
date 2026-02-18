from transformers import AutoTokenizer
from pathlib import Path
from typing import List
import json
from src.data.utils import vocabulary_path

_INSTANCE = None

class HFTokenizer:
    @classmethod
    def get_instance(cls, model_name: str):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls(model_name=model_name)
        return _INSTANCE
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.save_path = vocabulary_path / f"{model_name.replace('/', '_')}_vocab.json"
        
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]):
        return self.tokenizer.decode(token_ids)
    
    def save_vocabulary(self):
        with open(self.save_path, "w") as f:
            vocab = {v: k for k, v in self.tokenizer.vocab.items()}
            json.dump(vocab, f, indent=2, ensure_ascii=False)
            

if __name__ == "__main__":
    tokenizer = HFTokenizer.get_instance(model_name="TWO/sutra-mlt256-v2")
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    for token in tokens:
        print(tokenizer.decode([token]).strip(" "), token)
        
    tokenizer.save_vocabulary()
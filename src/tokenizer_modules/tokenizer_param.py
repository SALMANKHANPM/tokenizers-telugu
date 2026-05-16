from transformers import AutoTokenizer
from typing import List
from src.data.utils import vocabulary_path
import json
import os

class ParamTokenizer:
    _instances: dict = {}

    @classmethod
    def get_instance(cls, model_path: str):
        if model_path not in cls._instances:
            cls._instances[model_path] = cls(model_path=model_path)
        return cls._instances[model_path]

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,  trust_remote_code=True)
        #self.save_path = vocabulary_path / f"{model_path.name.rstrip("tokenizer.model")}vocab.json"
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.tokenizer.vocab_size}, backend={type(self.tokenizer).__name__})"
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(tokens, skip_special_tokens=skip_special_tokens) for tokens in batch]
    
    def save_vocabulary(self):
        model_name = self.tokenizer.name_or_path.split('/')[-1]
        save_path = vocabulary_path / f"{model_name}_vocab.json"
        os.makedirs(vocabulary_path, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({self.tokenizer.decode([v]): v for k, v in self.tokenizer.vocab.items()}, f, indent=2, ensure_ascii=False)

        print(f"Vocabulary saved to {save_path}")
        
if __name__ == "__main__":
    #tokenizer = ParamTokenizer(model_path="bharatgenai/Param2-17B-A2.4B-Thinking")
    tokenizer = ParamTokenizer(model_path="TWO/sutra-mlt256-v2")
    str = "ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?"
    tokens = tokenizer.encode(str)
    print(tokens)
    for token in tokens:
        print(tokenizer.decode([token]), token)
    
    tokenizer.save_vocabulary()
    
    # tokenizer = ParamTokenizer(model_path="CohereLabs/tiny-aya-global")
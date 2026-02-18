import json
import os
from pathlib import Path
from typing import List, Union

from tiktoken import get_encoding, Encoding
from tiktoken.load import load_tiktoken_bpe
from tiktoken.model import MODEL_TO_ENCODING

from src.data.utils import vocabulary_path

_INSTANCE = None

MODEL_MAPPING = MODEL_TO_ENCODING.copy()
MODEL_MAPPING.update({"gpt-oss": "o200k_base"})


class OpenAITokenizer:
    @classmethod
    def get_instance(cls, model_or_path: Union[str, Path] = "gpt-4o") -> "OpenAITokenizer":
        global _INSTANCE
        cache_key = str(model_or_path)
        
        if _INSTANCE is None or getattr(_INSTANCE, 'cache_key', None) != cache_key:
            _INSTANCE = cls(model_or_path=model_or_path)
        return _INSTANCE

    def __init__(self, model_or_path: Union[str, Path]):
        self.cache_key = str(model_or_path)
        
        # Check if it's a Path object or file path string
        if isinstance(model_or_path, Path) or (isinstance(model_or_path, str) and os.path.exists(model_or_path)):
            # Load from file path using load_tiktoken_bpe
            path = Path(model_or_path)
            self.model_name = path.stem.replace("_tokenizer", "")
            
            # Load the BPE file
            mergeable_ranks = load_tiktoken_bpe(str(path))
            
            # Determine special tokens based on model name
            if "o200k" in self.model_name or "o1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name or "gpt-5" in self.model_name:
                special_tokens = {
                    "<|endoftext|>": 200000,
                    "<|endofprompt|>": 200001,
                }
            elif "cl100k" in self.model_name:
                special_tokens = {
                    "<|endoftext|>": 100257,
                    "<|fim_prefix|>": 100258,
                    "<|fim_middle|>": 100259,
                    "<|fim_suffix|>": 100260,
                    "<|endofprompt|>": 100261,
                }
            else:
                special_tokens = {"<|endoftext|>": len(mergeable_ranks)}
            
            # Create encoding manually
            self.tokenizer = Encoding(
                name=self.model_name,
                pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens,
            )
        else:
            # Use model name to get encoding
            self.model_name = str(model_or_path)
            encoding = self.model_to_encoding(model_name=self.model_name)
            if encoding is None:
                available_models = ", ".join(MODEL_MAPPING.keys())
                print(f"Available models: {available_models}")
                raise ValueError(f"Unknown model name: {self.model_name}\nAvailable models: {available_models}")
            self.tokenizer = get_encoding(encoding)
        
        self.save_path = vocabulary_path / f"{self.model_name}_vocab.json"
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, vocab_size={self.tokenizer.n_vocab})"

    @staticmethod
    def model_to_encoding(model_name: str) -> str | None:
        return MODEL_MAPPING.get(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def get_vocabulary(self):
        token_bytes = self.tokenizer.token_byte_values()  
        token_texts = [b.decode('utf-8', errors='replace') for b in token_bytes]  
        vocabulary = {i: token_texts[i] for i in range(len(token_bytes))}
        return vocabulary
    
    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    tokenizer = OpenAITokenizer.get_instance(model_name="o1")
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    for token in tokens:
        print(tokenizer.decode([token]), token)
        
    tokenizer.save_vocabulary()
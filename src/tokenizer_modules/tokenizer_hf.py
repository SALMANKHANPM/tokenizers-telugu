import json
from typing import List

from transformers import AutoTokenizer

from src.data.utils import vocabulary_path


class HFTokenizer:
    _instances: dict = {}

    @classmethod
    def get_instance(cls, model_name: str):
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name=model_name)
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        self.save_path = vocabulary_path / f"{model_name.replace('/', '_')}_vocab.json"

    def __repr__(self) -> str:
        try:
            backend = type(self.tokenizer.backend_tokenizer.model).__name__
        except AttributeError:
            backend = type(self.tokenizer).__name__
        return f"{self.__class__.__name__}(model={self.tokenizer.name_or_path}, vocab_size={self.tokenizer.vocab_size}, tokenizer_backend={backend})"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch
        ]

    def get_vocabulary(self) -> dict[int, str]:
        return {
            i: self.decode([i], skip_special_tokens=False)
            for i in range(self.tokenizer.vocab_size)
        }

    def save_vocabulary(self):
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {self.save_path}")


if __name__ == "__main__":
    tokenizer = HFTokenizer.get_instance(
        model_name="krutrim-ai-labs/Krutrim-2-instruct"
    )
    print(tokenizer)
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    for token in tokens:
        print(tokenizer.decode([token]).strip(" "), token)

    tokenizer.save_vocabulary()

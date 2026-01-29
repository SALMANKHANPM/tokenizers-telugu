from typing import List

from tiktoken import get_encoding
from tiktoken.model import MODEL_TO_ENCODING

_INSTANCE = None

MODEL_MAPPING = MODEL_TO_ENCODING.copy()
MODEL_MAPPING.update({"gpt-oss": "o200k_base"})


class OpenAITokenizer:
    @classmethod
    def get_instance(cls, model_name: str = "gpt-4o") -> "OpenAITokenizer":
        global _INSTANCE
        if _INSTANCE is None or _INSTANCE.model_name != model_name:
            _INSTANCE = cls(model_name=model_name)
        return _INSTANCE

    def __init__(self, model_name: str):
        self.model_name = model_name
        encoding = self.model_to_encoding(model_name=model_name)
        if encoding is None:
            available_models = ", ".join(MODEL_MAPPING.keys())
            print(f"Available models: {available_models}")
            raise ValueError(f"Unknown model name: {model_name}\nAvailable models: {available_models}")
        self.tokenizer = get_encoding(encoding)

    @staticmethod
    def model_to_encoding(model_name: str) -> str | None:
        return MODEL_MAPPING.get(model_name)


    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)


if __name__ == "__main__":
    tokenizer = OpenAITokenizer.get_instance(model_name="o1")
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    for token in tokens:
        print(tokenizer.decode([token]), token)
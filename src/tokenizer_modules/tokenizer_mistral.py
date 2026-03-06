import json
import os
from pathlib import Path
from typing import List
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from src.data.utils import vocabulary_path

class MistralTokenizer:
    _instance = None

    @classmethod
    def get_instance(cls, path: Path):
        if cls._instance is None:
            cls._instance = cls(model_path=path)
        return cls._instance
    
    def __init__(self, model_path: Path):
        self.tokenizer = SentencePieceTokenizer(model_path)
        self.save_path = vocabulary_path / f"{model_path.stem.replace('_tokenizer', '')}_vocab.json"
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.tokenizer.n_words}, backend={type(self.tokenizer).__name__})"

    def encode(self, s: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(s, bos=add_special_tokens, eos=add_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(t, skip_special_tokens=skip_special_tokens) for t in batch]

    def decode(self, t: List[int], skip_special_tokens: bool = True) -> str:
        policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP
        return self.tokenizer.decode(t, special_token_policy=policy)
    
    def get_vocabulary(self):
        return {i: self.decode([i], skip_special_tokens=False) for i in range(self.tokenizer.n_words)}
    
    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)  
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {self.save_path}")
    
class MistralTekkenizer:
    _instance = None

    @classmethod
    def get_instance(cls, path: Path):
        if cls._instance is None:
            cls._instance = cls(model_path=path)
        return cls._instance
    
    def __init__(self, model_path: Path):
        self.tokenizer = Tekkenizer.from_file(model_path)
        self.save_path = vocabulary_path / f"{model_path.stem.replace('_tokenizer', '')}_vocab.json"
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.tokenizer.n_words}, backend={type(self.tokenizer).__name__})"

    def encode(self, s: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(s, bos=add_special_tokens, eos=add_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(t, skip_special_tokens=skip_special_tokens) for t in batch]

    def decode(self, t: List[int], skip_special_tokens: bool = True) -> str:
        policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP
        return self.tokenizer.decode(t, special_token_policy=policy)

    def get_vocabulary(self) -> dict[int, str]:
        return {i: self.decode([i], skip_special_tokens=False) for i in range(self.tokenizer.n_words)}
    
    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {self.save_path}")
    

if __name__ == "__main__":
    # tekken_tokenizer = MistralTekkenizer.get_instance(path=Path(__file__).parent.parent / "tokenizer-models" / "mistral_small_tekken" / "mistral_tekken_tokenizer.json")
    # tokens = tekken_tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    # for token in tokens:
    #     print(tekken_tokenizer.decode([token]), token)
    # tekken_tokenizer.save_vocabulary()
    
    sp_tokenizer = MistralTokenizer.get_instance(path=Path(__file__).parent.parent / "tokenizer-models" / "mistral_sp" / "mistral_sp_tokenizer.model")
    tokens_sp = sp_tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    for token in tokens_sp:
        print(sp_tokenizer.decode([token]), token)
    sp_tokenizer.save_vocabulary()

    print(tokens_sp == tokens_sp)
    
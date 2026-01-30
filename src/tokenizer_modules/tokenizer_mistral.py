import json
import os
from pathlib import Path
from typing import List
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from src.data.utils import vocabulary_path

_INSTANCE = None

class MistralTokenizer:
    @classmethod
    def get_instance(cls, path: Path):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls(model_path=path)
        return _INSTANCE
    
    def __init__(self, model_path: Path):
        self.tokenizer = SentencePieceTokenizer(model_path)
        self.save_path = vocabulary_path / f"{model_path.name.rstrip("tokenizer.model")}vocab.json"
        
    def encode(self, s: str, bos: bool = True, eos: bool = True) -> List[int]:
        return self.tokenizer.encode(s, bos=bos, eos=eos)
    
    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t, special_token_policy=SpecialTokenPolicy.KEEP)
    
    def get_vocabulary(self):
        return {i: self.decode([i]) for i in range(self.tokenizer.n_words)}
    
    def save_vocabulary(self):  
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
    
class MistralTekkenizer:
    @classmethod
    def get_instance(cls, path: Path):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls(model_path=path)
        return _INSTANCE
    
    def __init__(self, model_path: Path):
        self.tokenizer = Tekkenizer.from_file(model_path)
        self.save_path = vocabulary_path / f"{model_path.name.rstrip("tokenizer.json")}vocab.json"
        
    def encode(self, s: str, bos: bool = True, eos: bool = True) -> List[int]:
        return self.tokenizer.encode(s, bos=bos, eos=eos)
    
    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t, special_token_policy=SpecialTokenPolicy.KEEP)
    
    def get_vocabulary(self):
        vocabulary = {}
        for i in range(self.tokenizer.n_words):
            vocabulary[i] = self.tokenizer.decode([i], special_token_policy=SpecialTokenPolicy.KEEP)
        return vocabulary
    
    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
    

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
    
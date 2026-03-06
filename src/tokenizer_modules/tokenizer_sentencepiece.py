# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py
# https://github.com/google/gemma_pytorch/blob/main/gemma/tokenizer.py


import json
import os
from logging import getLogger
from pathlib import Path
from typing import List

from sentencepiece import SentencePieceProcessor

from src.data.utils import vocabulary_path


logger = getLogger()

class SentencePieceTokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    _instances: dict = {}

    @classmethod
    def get_instance(cls, model_path: Path):
        key = str(model_path)
        if key not in cls._instances:
            cls._instances[key] = cls(model_path=model_path)
        return cls._instances[key]
    
    def __init__(self, model_path: Path):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        self.sp_model = SentencePieceProcessor(model_file=str(model_path))
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.save_path = vocabulary_path / f"{model_path.stem.replace('_tokenizer', '')}_vocab.json"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.n_words}, backend={type(self.sp_model).__name__})"

    def encode(self, s: str, add_special_tokens: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if add_special_tokens:
            t = [self.bos_id] + t + [self.eos_id]
        return t

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode_batch(self, batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(t, skip_special_tokens=skip_special_tokens) for t in batch]

    def decode(self, t: List[int], skip_special_tokens: bool = True) -> str:
        return self.sp_model.decode(t)

    def get_vocabulary(self) -> dict[int, str]:
        return {i: self.decode([i], skip_special_tokens=False) for i in range(self.n_words)}

    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {self.save_path}")
if __name__ == "__main__":
    # Usage
    tokenizer = SentencePieceTokenizer.get_instance(model_path=Path(__file__).parent.parent / "tokenizer-models" / "sarvam1" / "sarvam1_tokenizer.model")
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి")
    for token in tokens:
        print(tokenizer.decode([token]).strip(" "), token)
        
    tokenizer.save_vocabulary()

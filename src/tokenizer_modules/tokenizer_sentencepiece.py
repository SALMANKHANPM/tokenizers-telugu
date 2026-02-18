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

_INSTANCE = None

class SentencePieceTokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    @classmethod
    def get_instance(cls, model_path: Path):
        global _INSTANCE

        if _INSTANCE is None:
            _INSTANCE = SentencePieceTokenizer(model_path=model_path)
        return _INSTANCE
    
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
        self.save_path = vocabulary_path / f"{model_path.name.strip("tokenizer.model")}vocab.json"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.n_words})"

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)

    def get_vocabulary(self) -> dict[int, str]:
        vocabulary = {}
        for i in range(self.sp_model.get_piece_size()):
            vocabulary[i] = self.sp_model.id_to_piece(i)
        return vocabulary

    def save_vocabulary(self):
        os.makedirs(self.save_path.parent, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.get_vocabulary(), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Usage
    tokenizer = SentencePieceTokenizer.get_instance(model_path=Path(__file__).parent.parent / "tokenizer-models" / "sarvam1" / "sarvam1_tokenizer.model")
    tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి", bos=False, eos=False)
    for token in tokens:
        print(tokenizer.decode([token]).strip(" "), token)
        
    tokenizer.save_vocabulary()

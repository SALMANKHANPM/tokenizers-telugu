import json
from src.data.models import model_id
from typing import List

class TokenizerFactory:
    @classmethod
    def get_tokenizer(cls, model_name: str):
        for model, model_path, tokenizer_class in model_id:
            if model_name == model:
                print("Loading tokenizer for model: ", model_name)
                print("Model path: ", model_path)
                print("Tokenizer class: ", tokenizer_class.__name__)
                return tokenizer_class.get_instance(model_path)
        raise ValueError(f"Tokenizer not found for model: {model_name}")
    
    def __init__(self, model_name: str):
        self.tokenizer = self.get_tokenizer(model_name)
        self.tokenizers = []
        
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]):
        return self.tokenizer.decode(token_ids)
    
    
    @classmethod
    def get_all_tokenizers(cls):
        cls.tokenizers = [tokenizer_class.get_instance(model_path) for model, model_path, tokenizer_class in model_id]
        return cls.tokenizers
    
    def save_vocabulary(self):
        with open(self.save_path, "w") as f:
            json.dump(self.tokenizer.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def save_all_vocabulary(cls):
        for tokenizer in cls.get_all_tokenizers():
            tokenizer.save_vocabulary()
    
if __name__ == "__main__":
    # TokenizerFactory.get_all_tokenizers()
    # tokenizer = TokenizerFactory.get_tokenizer(model_name="mistral_sp")
    # tokenizer.save_all_vocabulary()
    # tokens = tokenizer.encode("ఎలా టైపు చెయ్యాలో వివరంగా తెలుసుకోండి, Hello, how are you?")
    # for token in tokens:
    #     print(tokenizer.decode([token]), token)
    
    TokenizerFactory.save_all_vocabulary()
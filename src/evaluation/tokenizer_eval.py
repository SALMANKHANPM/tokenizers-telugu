import json
from datatrove.utils.word_tokenizers import load_word_tokenizer
from tokenizers import Tokenizer
from ..data.languages import language_codes
from ..data.models import model_id
from datasets import load_dataset, Dataset
from tqdm import tqdm
from typing import List, Dict

class TokenizerEvaluator:
    def __init__(self, model_name: str):
        self.models = model_id
        self.language_codes = language_codes
        self.dataset = None
        self.results = []
        self.word_tokens = []
        self.tokens = []
            
    def load_data(self, lang_code: str):
        return load_dataset("salmankhanpm/tokenizer-eval-set-raw", lang_code, split="train")
    
    def tokenize(self, lang_id: str, tokenizer: Tokenizer):
        word_tokenizer = load_word_tokenizer(lang_id)
        for sample in tqdm(self.dataset['text'], desc=f"Tokenizing {lang_id}"):
            word_tokens = word_tokenizer.word_tokenize(sample)
            self.word_tokens.append(word_tokens)
            tokens = tokenizer.encode(sample).ids
            self.tokens.append(tokens)
        return self.word_tokens, self.tokens
    
    def compute_metrics(self, word_tokens: List[List[str]], tokens: List[List[int]]):
        for word_token, token in zip(word_tokens, tokens):
            pcw = len(word_token) / len(token)
            pct = len(token) / len(word_token)
            return pcw, pct
    
    def save_results(self, results: List[Dict]):
        with open("results.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    

if __name__ == "__main__":
    evaluator = TokenizerEvaluator(model_name="mistral_sp")
    for lang_id, lang_code in language_codes:
        dataset = evaluator.load_data(lang_code)
        evaluator.tokenize(lang_id, evaluator.tokenizer)
        results = evaluator.compute_metrics(evaluator.word_tokens, evaluator.tokens)
        evaluator.save_results(results)
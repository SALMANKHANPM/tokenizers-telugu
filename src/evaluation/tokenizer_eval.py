from collections import defaultdict
import json
from datasets import load_dataset
from datatrove.utils.word_tokenizers import WordTokenizer, load_word_tokenizer
from src.data.models import model_id
from src.data.languages import language_codes
import numpy as np

class TokenizerEvaluator:
    def __init__(self, dataset_id: str, language_codes: list[tuple[str, str]], model_id: list[tuple[str, str, str]]):
        self.dataset_id = dataset_id
        self.language_codes = language_codes
        self.model_id = model_id
        self.results = defaultdict(list)

    def load_model(self, model_id: str):
        for model, model_path, tokenizer_class in model_id:
            if model_id == model:
                return tokenizer_class.get_instance(model_path)
        raise ValueError(f"Model not found for model_id: {model_id}")
    
    def compute_metrics(self, text: str, word_tokenizer: WordTokenizer, tokenizer):
        words = word_tokenizer.word_tokenize(text)
        tokens = tokenizer.encode_batch(words)
        tokens_per_word = np.array(list(map(len, tokens)))
        fertility = np.mean(tokens_per_word).item()
        pcw = float((tokens_per_word >= 2).sum() / len(tokens_per_word))
        
        return len(words), int(np.sum(tokens_per_word)), fertility, pcw
    
    def process(self):
        for lang_id, lang_code in self.language_codes:
            dataset = load_dataset(self.dataset_id, lang_code, split="train")
            if dataset.num_rows < 100:
                continue
            ds_iter = iter(dataset)
            text = "\n".join([next(ds_iter)["text"] for _ in range(100)])
            word_tokenizer = load_word_tokenizer(lang_id)
            print(f"------------{lang_id} | {lang_code}--------------------")
            for model, model_path, tokenizer_class in self.model_id:
                tokenizer = tokenizer_class.get_instance(model_path)
                metrics = self.compute_metrics(text, word_tokenizer, tokenizer)
                
                print(f"Model: {model}, Text: {text[:10]}, Tokenizer: {tokenizer}")
                print(metrics)
                
                self.results[lang_id].append({
                    "model": model,
                    "words_count": metrics[0],
                    "tokens_count": metrics[1],
                    "fertility": metrics[2],
                    "pcw": metrics[3]
                })
            
        return self.results
    
    def save_results(self, results: dict):
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    tokenizer_evaluator = TokenizerEvaluator(dataset_id="salmankhanpm/tokenizer-eval-set", language_codes=language_codes, model_id=model_id)
    results = tokenizer_evaluator.process()
    tokenizer_evaluator.save_results(results)
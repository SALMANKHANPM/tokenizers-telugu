from datasets import load_dataset
from transformers import AutoTokenizer
from datatrove.utils.word_tokenizers import load_word_tokenizer
import pandas as pd
import numpy as np

tokenizers = [
    ("Llama3", "meta-llama/Llama-3.2-1B"),
    ("Gemma3", "google/gemma-3-1b-pt"),
    ("Mistral (S)", "mistralai/Mistral-Small-24B-Instruct-2501"),
    ("Qwen3", "Qwen/Qwen3-4B")
]

languages = [
    ("English", "eng_Latn", "en"),
    ("Chinese", "cmn_Hani", "zh"),
    ("French", "fra_Latn", "fr"),
    ("Arabic", "arb_Arab", "ar"),
]


wikis = {}
for lang_name, lang_code, short_lang_code in languages:
	wiki_ds = load_dataset("wikimedia/wikipedia", f"20231101.{short_lang_code}", streaming=True, split="train")
	wiki_ds = wiki_ds.shuffle(seed=42, buffer_size=10_000)
	
	ds_iter = iter(wiki_ds)
	wikis[lang_code] = "\n".join([next(ds_iter)["text"] for _ in range(100)])

results = []

def compute_tokenizer_metrics(tokenizer, word_tokenizer, text):
    """
    Computes fertility and proportion of continued words.
    
    Returns:
        tuple: (fertility, proportion_continued_words)
            - fertility: average tokens per word (lower is better)
            - proportion_continued_words: percentage of words split into 2+ tokens (lower is better)

    """
    words = word_tokenizer.word_tokenize(text)
    print("words", words[0:10])
    tokens = tokenizer.encode(words, add_special_tokens=False)
    print("tokens", tokens[0:10])
    tokens_per_word = np.array(list(map(len, tokens["input_ids"])))
    
    fertility = np.mean(tokens_per_word).item()
    proportion_continued_words = (tokens_per_word >= 2).sum() / len(tokens_per_word)
    
    return fertility, proportion_continued_words

for tokenizer_name, tokenizer_path in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    for lang_name, lang_code, short_lang_code in languages:
        word_tokenizer = load_word_tokenizer(lang_code)
        
        # Compute metrics on Wikipedia
        fertility, pcw = compute_tokenizer_metrics(tokenizer, word_tokenizer, wikis[lang_code])
        
        results.append({
            "tokenizer": tokenizer_name,
            "language": lang_name,
            "fertility": fertility,
            "pcw": pcw
        })

df = pd.DataFrame(results)
print(df)


#=========================
import sys
from pathlib import Path

# Add project root to path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ..data.languages import language_codes
    from ..data.models import model_id
    from ..data.utils import tokenizer_eval_dataset_id as dataset_id
except ImportError:
    from src.data.languages import language_codes
    from src.data.models import model_id
    from src.data.utils import tokenizer_eval_dataset_id as dataset_id

from datasets import load_dataset  # noqa: E402
from typing import List, Dict  # noqa: E402
import json  # noqa: E402
from datatrove.utils.word_tokenizers import load_word_tokenizer  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402
from itertools import islice  # noqa: E402
from collections import defaultdict  # noqa: E402

class TokenizerEvaluator:
    def __init__(self):
        self.results = defaultdict(list)

    def process_dataset(self, dataset):
        # Take up to 100 samples, or all samples if fewer than 100
        samples = list(islice(dataset, 100))
        dataset_text = "\n".join([sample["text"] for sample in samples])
        return dataset_text
    
    def compute_metrics(self, text: str, word_tokenizer, tokenizer):
        word_tokens = word_tokenizer.word_tokenize(text)
        words_count = len(word_tokens)
        
        # Encode each word individually to count tokens per word
        tokens_per_word = []
        for word in word_tokens:
            word_token_ids = tokenizer.encode(word)
            tokens_per_word.append(len(word_token_ids))
        
        tokens_per_word = np.array(tokens_per_word)
        tokens_count = np.sum(tokens_per_word)
        
        fertility = np.mean(tokens_per_word).item()
        pcw = (tokens_per_word >= 2).sum() / len(tokens_per_word)
        
        return words_count, int(tokens_count), fertility, float(pcw)
    
    def process(self):
        for lang_id, lang_code in tqdm(language_codes, desc="Processing languages"):
            dataset =  load_dataset(dataset_id, lang_code, split="train")
            dataset_text = self.process_dataset(dataset)
            word_tokenizer = load_word_tokenizer(lang_id)
            for model_name, tokenizer_path, tokenizer_class in tqdm(model_id, desc="Processing at Model Level", leave=False):
                tokenizer = tokenizer_class.get_instance(tokenizer_path)
                words_count, tokens_count, fertility, pcw = self.compute_metrics(text=dataset_text, word_tokenizer=word_tokenizer, tokenizer=tokenizer)
                self.results[model_name].append({
                    "lang_id": lang_id,
                    "lang_code": lang_code,
                    "words_count": words_count,
                    "tokens_count": tokens_count,
                    "fertility": fertility,
                    "pcw": pcw
                })
        return self.results
    
    def save_results(self, results: List[Dict]):
        with open("results.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    evaluator = TokenizerEvaluator()
    results = evaluator.process()
    evaluator.save_results(results)

    print(results)
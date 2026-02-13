from src.lid.lid_mediapipe import LanguageDetector as MediaPipeLanguageDetector
from datatrove.utils.word_tokenizers import load_word_tokenizer
from string import punctuation
from tqdm import tqdm

from datasets import load_dataset, Dataset
dataset_id = "salmankhanpm/tokenizer-eval-set-raw"
lang_code = [  
    ("tel_Telu", "te"),  # Telugu [1](#5-0)   
    ("ben_Beng", "bn"),  # Bengali [2](#5-1)   
    ("guj_Gujr", "gu"),  # Gujarati [3](#5-2)   
    ("hin_Deva", "hi"),  # Hindi [4](#5-3)   
    ("kan_Knda", "kn"),  # Kannada [5](#5-4)   
    ("mal_Mlym", "ml"),  # Malayalam [6](#5-5)   
    ("mar_Deva", "mr"),  # Marathi [7](#5-6)   
    ("pan_Guru", "pa"),  # Punjabi (Panjabi) [8](#5-7)   
    ("tam_Taml", "ta"),  # Tamil [9](#5-8)   
    ("ory_Orya", "or"),  # Odia [10](#5-9)   
]

def load_data(subset : str = ""):
    dataset = load_dataset(dataset_id, subset, split="train")
    return dataset

def clean_data(detector):
    for tokenizer_lang_code, tgt_lang_code in lang_code:
        clean_samples = []
        dataset = load_data(subset=tgt_lang_code)
        word_tokenizer = load_word_tokenizer(tokenizer_lang_code)

        for sample in tqdm(dataset, desc=f"Cleaning {tgt_lang_code}"):
            text = sample["text"]
            tokens = word_tokenizer.word_tokenize(text)
            has_foreign_token = False

            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if token in punctuation or token.isdigit():
                    continue
                if token.isascii() and token.isalpha() and tgt_lang_code != "en":
                    has_foreign_token = True
                    break
                lang, prob = detector.detect(token)
                if prob < 0.75:
                    continue
                if lang != tgt_lang_code:
                    has_foreign_token = True
                    break

            if not has_foreign_token:
                clean_samples.append({"text": text})

        clean_dataset = Dataset.from_list(clean_samples)
        clean_dataset.push_to_hub(dataset_id.rstrip("-raw"), config_name=f"{tgt_lang_code}", split="train")

def main():
    detector = MediaPipeLanguageDetector.get_instance()
    clean_data(detector)
    detector.close()


if __name__ == "__main__":
    main()
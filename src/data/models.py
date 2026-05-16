from src.data.utils import model_path
from src.tokenizer_modules.tokenizer_hf import HFTokenizer
from src.tokenizer_modules.tokenizer_llama3 import Llama3Tokenizer
from src.tokenizer_modules.tokenizer_llama4 import Llama4Tokenizer
from src.tokenizer_modules.tokenizer_mistral import MistralTekkenizer, MistralTokenizer
from src.tokenizer_modules.tokenizer_olmo import OLMoTokenizer
from src.tokenizer_modules.tokenizer_openai import OpenAITokenizer
from src.tokenizer_modules.tokenizer_qwen import QWenTokenizer
from src.tokenizer_modules.tokenizer_sentencepiece import SentencePieceTokenizer

# tuple of (model_name, model_path, tokenizer class)
model_id = [
    ("gemma", model_path / "gemma" / "gemma_tokenizer.model", SentencePieceTokenizer),
    (
        "gemma3",
        model_path / "gemma3" / "gemma3_tokenizer.model",
        SentencePieceTokenizer,
    ),
    ("llama4", model_path / "llama4" / "llama4_tokenizer.model", Llama4Tokenizer),
    ("llama3", model_path / "llama3" / "llama3_tokenizer.model", Llama3Tokenizer),
    (
        "llama2",
        model_path / "llama2" / "llama2_tokenizer.model",
        SentencePieceTokenizer,
    ),
    (
        "ministral",
        model_path / "ministral" / "ministral_tekken_tokenizer.json",
        MistralTekkenizer,
    ),
    (
        "mistral_small_tekken",
        model_path / "mistral_small_tekken" / "mistral_tekken_tokenizer.json",
        MistralTekkenizer,
    ),
    (
        "mistral_sp",
        model_path / "mistral_sp" / "mistral_sp_tokenizer.model",
        MistralTokenizer,
    ),
    ("olmo", model_path / "olmo" / "olmo_tokenizer.json", OLMoTokenizer),
    ("qwen", model_path / "qwen" / "qwen_tokenizer.model", QWenTokenizer),
    (
        "sarvam1",
        model_path / "sarvam1" / "sarvam1_tokenizer.model",
        SentencePieceTokenizer,
    ),
    (
        "gpt-oss",
        model_path / "openai" / "o200k_base_tokenizer.tiktoken",
        OpenAITokenizer,
    ),
    (
        "o200k_base",
        model_path / "openai" / "o200k_base_tokenizer.tiktoken",
        OpenAITokenizer,
    ),  # o1,3,4 and gpt 5 uses 0200k_base
    (
        "cl100k_base",
        model_path / "openai" / "cl100k_base_tokenizer.tiktoken",
        OpenAITokenizer,
    ),
    (
        "p50k_base",
        model_path / "openai" / "p50k_base_tokenizer.tiktoken",
        OpenAITokenizer,
    ),
    ("tiny-aya-fire", "CohereLabs/tiny-aya-fire", HFTokenizer),
    ("TWO-sutra-mlt256", "TWO/sutra-mlt256-v2", HFTokenizer),
    ("Param2-17B-MOE", "bharatgenai/Param2-17B-A2.4B-Thinking", HFTokenizer),
    ("Krutrim-2", "krutrim-ai-labs/Krutrim-2-instruct", HFTokenizer),
    ("Sarvam-30B", "sarvamai/sarvam-30b", HFTokenizer),
    ("Phi-4-mini", "microsoft/Phi-4-mini-instruct", HFTokenizer),
    ("Phi-3.5 mini", "microsoft/Phi-3.5-mini-instruct", HFTokenizer),
    ("GLM-4.6V", "zai-org/GLM-4.6V-Flash", HFTokenizer),
    ("Kimi-VL-A3B-Thinking-2506", "moonshotai/Kimi-VL-A3B-Thinking-2506", HFTokenizer),
    ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", HFTokenizer),
    ("SmolLM 3", "HuggingFaceTB/SmolLM3-3B", HFTokenizer),
]

from tqdm import tqdm
import unicodedata
from collections import defaultdict
from pathlib import Path
import json
import os
from typing import Dict, List, Any


class UnicodeLanguageIdentifier:
    """
    Unicode-based token language identifier class.
    
    Policy:
    - Indic tokens (including dependent Mn marks) are classified directly
      into their language ISO buckets.
    - Arabic-script tokens are mapped to `ur-IN` by policy.
    - `invalid` is reserved only for true corruption (�, empty, control).
    - Emoji detection is Unicode-robust (not library-fragile).
    """
    
    def __init__(self):
        self.token_classes = defaultdict(list)
    
    def is_combining_mark(self, ch: str) -> bool:
        """Check if character is a combining mark."""
        return unicodedata.category(ch) == "Mn"

    def first_base_char(self, token: str):
        """Find first base character in token."""
        for ch in token:
            if not self.is_combining_mark(ch):
                return ch
        return None

    def is_emoji_like(self, ch: str) -> bool:
        """Check if character is emoji-like."""
        return unicodedata.category(ch) == "So" or 0x1F000 <= ord(ch) <= 0x1FAFF

    def unicode_token_identifier(self, token: str) -> str:
        """Identify language of a single token."""
        # ---------------- normalize ----------------
        if token.startswith("<<") and token.endswith(">>") and token.startswith("[") and token.endswith("]"):
            return "SPECIAL"
        
        token = token.removeprefix("\t")
        token = token.removesuffix("▁")
        token = token.removesuffix("Ġ")
        if not token:
            return "INVALID"
        token = token[-1]
        token = unicodedata.normalize("NFC", token)

        # ---------------- invalid ----------------
        if not token or not token.strip():
            return "INVALID"

        if "\ufffd" in token:
            return "INVALID"

        # ---------------- emoji ----------------
        if any(self.is_emoji_like(ch) for ch in token):
            return "EMOJI"

        # ---------------- reference character ----------------
        base = self.first_base_char(token)
        ref_char = base if base is not None else token[0]
        cp = ord(ref_char)
        cat = unicodedata.category(ref_char)

        # ---------------- control chars ----------------
        if cat.startswith("C"):
            return "INVALID"

        # ---------------- English ----------------
        if ref_char.isascii() and ref_char.isalpha():
            return "ENGLISH"

        # ---------------- Indic languages (ISO aligned) ----------------
        if 0x0C00 <= cp <= 0x0C7F:
            return "TELUGU"

        if 0x0B80 <= cp <= 0x0BFF:
            return "TAMIL"

        if 0x0C80 <= cp <= 0x0CFF:
            return "KANNADA"

        if 0x0D00 <= cp <= 0x0D7F:
            return "MALAYALAM"

        if 0x0A80 <= cp <= 0x0AFF:
            return "GUJARATI"

        if 0x0B00 <= cp <= 0x0B7F:
            return "ODIA"

        if 0x0A00 <= cp <= 0x0A7F:
            return "PUNJABI"

        if 0x1C50 <= cp <= 0x1C7F:
            return "SANTALI"

        if 0xABC0 <= cp <= 0xABFF:
            return "MEITEI"

        if 0x0900 <= cp <= 0x097F:
            return "HINDI"

        if 0x0980 <= cp <= 0x09FF:
            return "BENGALI"

        if 0x0600 <= cp <= 0x06FF:
            return "URDU"

        # ---------------- punctuation / symbols ----------------
        if cat.startswith("P"):
            first_cat = unicodedata.category(token[0]) if token else ""
            last_cat = unicodedata.category(token[-1]) if token else ""
            if first_cat.startswith("P") and not last_cat.startswith("P"):
                return "ENGLISH"
            return "PUNCTUATION"

        if cat.startswith("S"):
            return "SYMBOL"

        # ---------------- everything else ----------------
        return "NON-LETTER"

    def filter_tokens(self, vocab: Dict[str, Any]) -> Dict[str, List[str]]:
        """Filter vocabulary tokens by language."""
        self.token_classes = defaultdict(list)
        
        for token in tqdm(vocab.values()):
            from string import punctuation
            tokencp = token.strip(punctuation)
            if tokencp != "":
                label = self.unicode_token_identifier(tokencp)
            else:
                label = "INVALID"
            self.token_classes[label].append(token)

        return dict(self.token_classes)

    def save_filtered_vocabulary(self, vocab: Dict[str, Any], output_path: Path) -> None:
        """Filter and save vocabulary by language."""
        filtered = self.filter_tokens(vocab)
        
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        
        print(f"Total tokens: {len(vocab)}")
        print(f"Telugu tokens: {len(filtered.get('TELUGU', []))}")
        print(f"Filtered vocabulary saved to: {output_path}")


if __name__ == "__main__":
    """Main function to process vocabulary file."""
    vocab_path = Path("src/vocabulary/o200k_base_vocab.json")
    output_path = Path("src/vocabclasses/o200k_base_vocab_filtered.json")
    
    with open(vocab_path) as f:
        data = json.load(f)
    
    identifier = UnicodeLanguageIdentifier()
    identifier.save_filtered_vocabulary(data, output_path)

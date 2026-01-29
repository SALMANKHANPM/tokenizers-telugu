from tqdm import tqdm
import unicodedata
from collections import defaultdict
def unicode_token_identifier(token: str) -> str:
        """
        Final Unicode-based token language identifier.

        Policy:
        - Indic tokens (including dependent Mn marks) are classified directly
        into their language ISO buckets.
        - Arabic-script tokens are mapped to `ur-IN` by policy.
        - `invalid` is reserved only for true corruption (�, empty, control).
        - Emoji detection is Unicode-robust (not library-fragile).
        """

        # ---------------- helpers ----------------

        def is_combining_mark(ch: str) -> bool:
            return unicodedata.category(ch) == "Mn"

        def first_base_char(token: str):
            for ch in token:
                if not is_combining_mark(ch):
                    return ch
            return None

        def is_emoji_like(ch: str) -> bool:
            # Covers pictographs, flags, symbols
            return unicodedata.category(ch) == "So" or 0x1F000 <= ord(ch) <= 0x1FAFF

        # ---------------- normalize ----------------
        
        token = token.strip().strip("\n").strip("▁")
        
        token = unicodedata.normalize("NFC", token)

        # ---------------- invalid ----------------

        if not token or not token.strip():
            return "INVALID"

        if "\ufffd" in token:
            return "INVALID"

        # ---------------- special tokens ----------------

        if token.startswith("<<") and token.endswith(">>"
        ):
            return "SPECIAL"

        # ---------------- emoji ----------------

        if any(is_emoji_like(ch) for ch in token):
            return "EMOJI"

        # ---------------- reference character ----------------
        # Use first base letter if present, else first character (Mn-only case)
        base = first_base_char(token)
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
            return "TELUGU"  # Telugu

        if 0x0B80 <= cp <= 0x0BFF:
            return "TAMIL"  # Tamil

        if 0x0C80 <= cp <= 0x0CFF:
            return "KANNADA"  # Kannada

        if 0x0D00 <= cp <= 0x0D7F:
            return "MALAYALAM"  # Malayalam

        if 0x0A80 <= cp <= 0x0AFF:
            return "GUJARATI"  # Gujarati

        if 0x0B00 <= cp <= 0x0B7F:
            return "ODIA"  # Odia

        if 0x0A00 <= cp <= 0x0A7F:
            return "PUNJABI"  # Punjabi (Gurmukhi)

        if 0x1C50 <= cp <= 0x1C7F:
            return "SANTALI"  # Santali (Ol Chiki)

        if 0xABC0 <= cp <= 0xABFF:
            return "MEITEI"  # Meitei (Meetei Mayek)

        if 0x0900 <= cp <= 0x097F:
            return "HINDI"  # Devanagari → Hindi bucket (policy)

        if 0x0980 <= cp <= 0x09FF:
            return "BENGALI"  # Bengali script bucket (policy)

        if 0x0600 <= cp <= 0x06FF:
            return "URDU"  # Arabic script → Urdu bucket (policy)

        # ---------------- punctuation / symbols ----------------

        if cat.startswith("P"):
            # If token begins with punctuation but doesn't end with punctuation,
            # treat it as an English-ish token like ".iter", ",body", "});"
            first_cat = unicodedata.category(token[0]) if token else ""
            last_cat = unicodedata.category(token[-1]) if token else ""
            if first_cat.startswith("P") and not last_cat.startswith("P"):
                return "ENGLISH"
            return "PUNCTUATION"

        if cat.startswith("S"):
            return "SYMBOL"

        # ---------------- everything else ----------------

        return "NON-LETTER"

def filter_tokens(self):
        for token in tqdm(self.vocab):
            label = self.unicode_token_identifier(token)
            self.token_classes[label].append(token)

        return self.token_classes


def main():
    # args = ArgumentParser()
    # args.add_argument("--hf", help="Hugging Face model path")
    # args = args.parse_args()


    with open("./gemma2-token.txt") as f:
        data = f.readlines()
    
    ex = defaultdict(list)
    c = 0
    for token in data:
        token.strip("▁").strip(" ")
        label = unicode_token_identifier(token)
        ex[label].append(token)
        c +=1
        
    with open("./gemma2-token-filtered.json", "w") as f:
        import json
        json.dump(ex, f, indent=2, ensure_ascii=False)
        
    print(len(data), c)
    
    print(len(ex["TELUGU"]))
        

if __name__ == "__main__":
    main()

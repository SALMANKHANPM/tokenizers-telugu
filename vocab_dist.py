import json
from pathlib import Path

from tqdm import tqdm

from src.data.utils import vocabulary_path
from src.lid.lid_unicode import UnicodeLanguageIdentifier


class VocabDist:
    def __init__(self, vocab_path: Path):
        self.vocab_path = vocab_path
        self.filtered_vocab_path = Path("src/vocab-filtered")
        self.vocab = json.load(open(vocab_path))
        self.identifier = UnicodeLanguageIdentifier()
        self.filtered_vocab: dict[str, list[str]] = {}

    def filter_vocab(self) -> dict[str, list[str]]:
        for token in self.vocab.values():
            label = self.identifier.unicode_token_identifier(token)
            self.filtered_vocab.setdefault(label, []).append(token)
        return self.filtered_vocab

    def save_filtered_vocab(self):
        self.filtered_vocab_path.mkdir(parents=True, exist_ok=True)
        save_path = self.filtered_vocab_path / f"{self.vocab_path.stem}-filtered.json"
        with open(save_path, "w") as f:
            json.dump(self.filtered_vocab, f, indent=2, ensure_ascii=False)

        print(f"Filtered vocabulary saved to {save_path}")

    def print_summary(self):
        total = len(self.vocab)
        for label, tokens in sorted(
            self.filtered_vocab.items(), key=lambda x: -len(x[1])
        ):
            pct = len(tokens) / total * 100
            print(f"  {label:<15} {len(tokens):>7,}  ({pct:.1f}%)")
        print(f"  {'TOTAL':<15} {total:>7,}")


if __name__ == "__main__":
    for file in tqdm(sorted(vocabulary_path.absolute().iterdir())):
        if file.is_file() and file.suffix == ".json" and "filtered" not in file.name:
            print(f"\n=== {file.name} ===")
            vd = VocabDist(file)
            vd.filter_vocab()
            vd.save_filtered_vocab()
            vd.print_summary()

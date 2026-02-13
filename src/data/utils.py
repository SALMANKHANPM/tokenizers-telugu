from pathlib import Path

model_path = Path(__file__).parent.parent / "tokenizer-models"
vocabulary_path = Path(__file__).parent.parent / "vocabulary"
lid_model_path = Path(__file__).parent.parent / "lid-model" / "language_detector.tflite"
from pathlib import Path
from typing import Dict, List

from mediapipe.tasks import python
from mediapipe.tasks.python import text

from ..data.utils import lid_model_path

_INSTANCE = None

class LanguageDetector:

    @classmethod
    def get_instance(cls, model_path: Path = lid_model_path):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls(model_path=Path(model_path))
        return _INSTANCE

    @classmethod
    def close_instance(cls) -> None:
        global _INSTANCE
        if _INSTANCE is not None:
            _INSTANCE.close()
            _INSTANCE = None

    def __init__(self, model_path: Path):
        self.base_options = python.BaseOptions(model_asset_path=str(model_path))
        self.options = text.LanguageDetectorOptions(base_options=self.base_options)
        self.detector = text.LanguageDetector.create_from_options(self.options)
        self._closed = False

    def detect(self, input_text: str) -> List[Dict[str, float]]:
        if self._closed:
            raise RuntimeError("LanguageDetector is closed. Call get_instance() to create a new detector.")

        detection_result = self.detector.detect(input_text)

        detections = detection_result.detections[0].language_code
        probabilities = detection_result.detections[0].probability

        return detections, probabilities  # returns top 1 language and its probability : ('en', 0.9994996786117554)

    def close(self) -> None:
        if self._closed:
            return

        try:
            if self.detector is not None:
                # MediaPipe can throw during interpreter shutdown; close best-effort.
                self.detector.close()
        except (RuntimeError, TypeError):
            pass
        finally:
            self.detector = None
            self._closed = True

if __name__ == "__main__":
    print("Loading language detector...")
    path = lid_model_path
    print(path)
    detector = LanguageDetector.get_instance(path)
    print(detector.detect("Hello, how are you?"))
    detector.close()
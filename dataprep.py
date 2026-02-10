
from pathlib import Path
from src.lid.lid_mediapipe import LanguageDetector
from  src.data.utils import lid_model_path
from datasets import load_dataset, Dataset
from tqdm import tqdm

class LanguageDetectorFactory:
    @classmethod
    def get_language_detector(cls, model_path: Path):
        return LanguageDetector.get_instance(model_path)
    
    def __init__(self, model_path: Path):
        self.detector = LanguageDetector.get_instance(model_path)
        
    def detect(self, text: str):
        return self.detector.detect(text)
    
class DolciDataLanguageFilter:
    def __init__(self, language_detector: LanguageDetector):
        self.language_detector = language_detector
        self.dataset = None
        self.filtered_data = []
        
    def load_data(self, data_path: Path, local: bool = False, subset: str = "", split: str = "train"):
        if not local:
            self.dataset = load_dataset(data_path, subset, split=split)
        else:
            # TODO: Add support for local data
            #dataset = load_dataset("json", data_files=str(data_path / f"{subset}.json"))
            pass
        
        return self.dataset
    
    def filter_data(self):
        for sample in tqdm(self.dataset, desc="Filtering data"):
                user_content = sample["messages"][0]["content"]
                assistant_content = sample["messages"][1]["content"]
                
                combined_str = user_content + "\n" + assistant_content

                lang, _ = self.language_detector.detect(combined_str)
                if lang == "en":
                    self.filtered_data.append(sample)

        return self.filtered_data
    
    def upload_to_hf(self):
        print("Uploading to Hugging Face...")
        dataset = Dataset.from_list(self.filtered_data)
        dataset.push_to_hub("salmankhanpm/dolci-instruct-sft-no-tools-telugu", "ai2", split="train")
    
if __name__ == "__main__":    
    detector = LanguageDetectorFactory.get_language_detector(lid_model_path)
    filter = DolciDataLanguageFilter(detector)
    dataset = filter.load_data(data_path="salmankhanpm/dolci-instruct-sft-no-tools-telugu", local=False, subset="ai2", split="train")
    filtered_data = filter.filter_data()    
    filter.upload_to_hf()
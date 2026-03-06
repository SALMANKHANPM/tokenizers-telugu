from src.data.models import model_id

for model_name, model_path, tokenizer_class in model_id:
    tokenizer = tokenizer_class.get_instance(model_path)
    print(tokenizer)
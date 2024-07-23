from src.config import LIT_AVAILABLE, valid_labels

LIT_IMPORTS_SUCCESSFUL = False

if LIT_AVAILABLE:
    try:
        from lit_nlp.api import dataset as lit_dataset
        from lit_nlp.api import types as lit_types
        LIT_IMPORTS_SUCCESSFUL = True
    except ImportError:
        print("Не удалось импортировать модули LIT. Убедитесь, что LIT установлен корректно.")

class BertWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, inputs):
        encoded = self.tokenizer(inputs['text'], return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()
        return {'logits': logits}

    def input_spec(self):
        if not LIT_IMPORTS_SUCCESSFUL:
            raise ImportError("LIT не доступен. Невозможно использовать input_spec.")
        return {
            'text': lit_types.TextSegment(),
        }

    def output_spec(self):
        if not LIT_IMPORTS_SUCCESSFUL:
            raise ImportError("LIT не доступен. Невозможно использовать output_spec.")
        return {
            'logits': lit_types.MulticlassPreds(parent='text', vocab=valid_labels),
        }

def setup_lit(model, tokenizer, messages):
    if not LIT_AVAILABLE or not LIT_IMPORTS_SUCCESSFUL:
        print("LIT не доступен или не удалось импортировать необходимые модули. Невозможно настроить LIT.")
        return None, None

    wrapped_model = BertWrapper(model, tokenizer)
    
    # Создаем данные
    data = [{
        'text': msg,
        # Добавьте здесь другие поля, если они нужны
    } for msg in messages]
    
    # Создаем Dataset без явного указания спецификации
    dataset = lit_dataset.Dataset(data)
    
    # Используем модель напрямую без CachingModelWrapper
    return wrapped_model, dataset
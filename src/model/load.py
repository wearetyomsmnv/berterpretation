from transformers import BertTokenizer, BertForSequenceClassification
import time
import torch
from src.config import valid_labels

def load_model(model_path):
    print(f"Загрузка модели из {model_path}...")
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(valid_labels),
        ignore_mismatched_sizes=True
    )
    end_time = time.time()
    print(f"Модель загружена за {end_time - start_time:.2f} секунд")
    print(f"Размер модели: {sum(p.numel() for p in model.parameters()):,} параметров")
    print(f"Размер модели в памяти: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024):.2f} MB")
    print(f"Конфигурация модели:")
    print(f"  Количество слоев: {model.config.num_hidden_layers}")
    print(f"  Размер скрытого состояния: {model.config.hidden_size}")
    print(f"  Количество голов внимания: {model.config.num_attention_heads}")
    return tokenizer, model
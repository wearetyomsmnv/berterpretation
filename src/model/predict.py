import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import psutil
import os
from src.config import valid_labels
from captum.attr import IntegratedGradients

@torch.no_grad()
def predict_labels(message, tokenizer, model, device):
    start_time = time.time()
    
    max_length = min(128, max(len(message.split()) + 2, 16))
    
    print(f"[DEBUG] Длина входного сообщения: {len(message.split())} слов")
    print(f"[DEBUG] Используемая max_length: {max_length}")
    
    tokenization_start = time.time()
    inputs = tokenizer.encode_plus(
        message,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)
    tokenization_end = time.time()
    
    print(f"[DEBUG] Время токенизации: {tokenization_end - tokenization_start:.4f} секунд")
    print(f"[DEBUG] Форма входных данных: {inputs['input_ids'].shape}")
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"[DEBUG] Первые 10 токенов: {tokens[:10]}")
    print(f"[DEBUG] Последние 10 токенов: {tokens[-10:]}")
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            outputs = model(**inputs)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    predictions = torch.sigmoid(outputs.logits).squeeze()
    
    threshold = 0.5
    predicted_labels = [valid_labels[i] for i in range(len(valid_labels)) if predictions[i] > threshold]
    
    important_tokens = {}
    
    if len(predicted_labels) > 0:
        def forward_func(embeddings):
            return model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask']).logits
        
        embeddings = model.bert.embeddings.word_embeddings(inputs['input_ids'])
        ig = IntegratedGradients(forward_func)
        
        for label in predicted_labels:
            label_index = valid_labels.index(label)
            attributions, delta = ig.attribute(inputs=embeddings,
                                               target=label_index,
                                               n_steps=20,
                                               return_convergence_delta=True)
            
            print(f"[DEBUG] Convergence delta для метки {label}: {delta}")
            
            token_importance = attributions.sum(dim=-1).squeeze(0)
            top_indices = token_importance.argsort(descending=True)[:5]
            important_tokens[label] = {
                'tokens': [tokens[i] for i in top_indices if tokens[i] not in ('[CLS]', '[SEP]', '[PAD]')],
                'score': predictions[label_index].item()
            }
    
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Предсказание выполнено за {inference_time:.4f} секунд")
    print(f"Скорость обработки: {1/inference_time:.2f} сообщений в секунду")
    
    process = psutil.Process(os.getpid())
    print(f"[DEBUG] Использование памяти: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"[DEBUG] Использование CUDA памяти: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"[DEBUG] Максимальное использование CUDA памяти: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    
    return predicted_labels, important_tokens, predictions.cpu().numpy()
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients
import torch

def visualize_attention(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention = model(input_ids, output_attentions=True)
    
    # Проверяем, есть ли выход внимания
    if attention.attentions is None or len(attention.attentions) == 0:
        print("Внимание не может быть визуализировано: модель не вернула слои внимания.")
        return
    
    # Берем последний слой внимания
    last_layer_attention = attention.attentions[-1]
    
    # Проверяем размерность
    if last_layer_attention.dim() < 3:
        print("Внимание не может быть визуализировано: неправильная размерность.")
        return
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    try:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_attentions=True)
            
            attention = outputs.attentions[-1]  # Берем внимание из последнего слоя
            
            # Берем внимание для первого токена ([CLS]) по отношению ко всем остальным
            attention_cls = attention[0, 0, 0, :].detach().cpu().numpy()
            
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(tokens)), attention_cls)
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.title("Attention from [CLS] token")
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Ошибка при визуализации внимания: {e}")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Детали ошибки: {str(e)}")
        import traceback
        traceback.print_exc()

def interpret_prediction_captum(model, tokenizer, sentence, target_label_idx):
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id

    def forward_func(inputs):
        return model(inputs)[0]

    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

    encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"]

    ref_input_ids = torch.tensor([[cls_token_id] + [ref_token_id] * (input_ids.shape[1] - 2) + [sep_token_id]])

    try:
        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=ref_input_ids,
                                            target=target_label_idx,
                                            return_convergence_delta=True)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        # Визуализация
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(tokens)), attributions)
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.title("Token Attributions")
        plt.tight_layout()
        plt.show()

        for token, attribution in zip(tokens, attributions):
            print(f"{token}: {attribution}")
    except Exception as e:
        print(f"Ошибка при интерпретации с помощью Captum: {e}")
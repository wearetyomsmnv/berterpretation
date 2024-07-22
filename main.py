import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from colorama import init, Fore, Back, Style as ColoramaStyle
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import time
import psutil
import os
import gc
from torch.profiler import profile, record_function, ProfilerActivity
import random
import matplotlib.pyplot as plt
import platform
import pyfiglet
from termcolor import colored

# Попытка импорта LIT
try:
    from lit_nlp.api import dataset as lit_dataset
    from lit_nlp.api import types as lit_types
    from lit_nlp.lib import caching
    from lit_nlp.dev_server import Server
    LIT_AVAILABLE = True
except ImportError:
    print("WARNING: LIT (Language Interpretability Tool) не может быть импортирован. Функциональность LIT будет отключена.")
    LIT_AVAILABLE = False

# Инициализация colorama для Windows
init()

def get_username():
    try:
        if platform.system() == "Windows":
            return os.getenv("USERNAME")
        else:
            return os.getenv("USER")
    except:
        return "User"

def print_banner(username):
    # Создаем текст баннера
    banner_text = f"Hello, {username}!"
    
    # Генерируем ASCII-арт из текста
    ascii_art = pyfiglet.figlet_format(banner_text, font="slant")
    
    # Добавляем цвет
    colored_ascii = colored(ascii_art, "cyan")
    
    # Создаем рамку
    width = len(ascii_art.splitlines()[0])
    horizontal_border = colored("+" + "=" * (width + 2) + "+", "yellow")
    
    # Выводим баннер
    print(horizontal_border)
    for line in colored_ascii.splitlines():
        print(colored("| ", "yellow") + line + colored(" |", "yellow"))
    print(horizontal_border)

valid_labels = ["about", "........ добавьте свои"]

# Расширенная карта цветов для меток
color_map = {
    'humor': Fore.CYAN,
    'своя_метка': Fore.YELLOW,
    'своя_метка': Fore.GREEN,
    'своя_метка': Fore.LIGHTGREEN_EX,
    'своя_метка': Fore.LIGHTBLUE_EX,
    'своя_метка': Fore.LIGHTCYAN_EX,
    'своя_метка': Fore.LIGHTYELLOW_EX,
    'своя_метка': Fore.LIGHTMAGENTA_EX,
    'своя_метка': Fore.LIGHTRED_EX,
    'своя_метка': Fore.LIGHTWHITE_EX,
    'своя_метка': Fore.MAGENTA,
    'своя_метка': Fore.LIGHTMAGENTA_EX,
    'своя_метка': Fore.RED,
    'своя_метка': Fore.LIGHTRED_EX,
    'своя_метка': Fore.BLUE,
    'своя_метка': Fore.LIGHTBLUE_EX,
    'своя_метка': Fore.CYAN,
    'своя_метка': Fore.LIGHTCYAN_EX,
    'своя_метка': Fore.BLUE,
    'своя_метка': Fore.WHITE,
    'своя_метка': Fore.RED,
    'своя_метка': Fore.LIGHTRED_EX
}

# Добавим случайные цвета для меток, которые не имеют фиксированного цвета
for label in valid_labels:
    if label not in color_map:
        color_map[label] = random.choice([Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTMAGENTA_EX])

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

def get_color(label):
    return color_map.get(label, Fore.WHITE)

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

def get_colored_labels(labels, important_tokens):
    return [f"{Back.WHITE}{get_color(label)}{label} ({important_tokens.get(label, {'score': 0})['score']:.2%}){ColoramaStyle.RESET_ALL}" for label in labels]

def highlight_words(message, important_tokens):
    words = message.split()
    highlighted_words = []
    for word in words:
        highlighted = False
        for label, info in important_tokens.items():
            if any(token.lower() in word.lower() for token in info['tokens']):
                color = get_color(label)
                highlighted_words.append(f"{color}{word}{ColoramaStyle.RESET_ALL}")
                highlighted = True
                break
        if not highlighted:
            highlighted_words.append(word)
    return ' '.join(highlighted_words)

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
        return {
            'text': lit_types.TextSegment(),
        }

    def output_spec(self):
        return {
            'logits': lit_types.MulticlassPreds(parent='text', vocab=valid_labels),
        }

def setup_lit(model, tokenizer, messages):
    if not LIT_AVAILABLE:
        print("LIT не доступен. Невозможно настроить LIT.")
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

def interactive_mode(tokenizer, model, device):
    style = Style.from_dict({
        'prompt': '#ansigreen bold',
        'input': '#ansiblue',
    })
    session = PromptSession(style=style)

    print("Интерактивный режим. Введите 'exit' для выхода.")
    total_time = 0
    total_messages = 0
    while True:
        try:
            message = session.prompt("Введите сообщение: ", style=style)
            if message.lower() == 'exit':
                break
            
            start_time = time.time()
            predicted_labels, important_tokens, all_predictions = predict_labels(message, tokenizer, model, device)
            end_time = time.time()
            
            total_time += end_time - start_time
            total_messages += 1
            
            colored_labels = get_colored_labels(predicted_labels, important_tokens)
            highlighted_message = highlight_words(message, important_tokens)
            print(f"Сообщение: {highlighted_message}")
            print(f"Предсказанные метки: {' '.join(colored_labels)}")
            print("Вероятности всех меток:")
            for label, prob in zip(valid_labels, all_predictions):
                print(f"  {get_color(label)}{label}: {prob:.2%}{ColoramaStyle.RESET_ALL}")
            print(f"Важные токены для каждой метки:")
            for label, info in important_tokens.items():
                print(f"  {get_color(label)}{label}: {', '.join(info['tokens'])}{ColoramaStyle.RESET_ALL}")
            
            print("\n[DEBUG] Статистика предсказаний:")
            print(f"  Количество предсказанных меток: {len(predicted_labels)}")
            print(f"  Средняя вероятность предсказанных меток: {np.mean([info['score'] for info in important_tokens.values()]):.2%}")
            print(f"  Максимальная вероятность: {np.max(all_predictions):.2%}")
            print(f"  Минимальная вероятность: {np.min(all_predictions):.2%}")
            print(f"  Время обработки: {end_time - start_time:.4f} секунд")
            print(f"  Среднее время обработки: {total_time / total_messages:.4f} секунд")
            
            print("\nПопытка визуализации внимания...")
            try:
                visualize_attention(model, tokenizer, message)
            except Exception as e:
                print(f"Ошибка при визуализации внимания: {e}")

            
            if predicted_labels:
                print("\nПопытка интерпретации предсказания с помощью Captum...")
                try:
                    target_label_idx = valid_labels.index(predicted_labels[0])
                    interpret_prediction_captum(model, tokenizer, message, target_label_idx)
                except Exception as e:
                    print(f"Ошибка при интерпретации с помощью Captum: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except KeyboardInterrupt:
            print("\nВыход из интерактивного режима.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")

def main():
    parser = argparse.ArgumentParser(description="Классификация текста с использованием BERT")
    parser.add_argument("--model", type=str, default="model", help="Путь к модели")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Устройство для вычислений (cuda/cpu)")
    parser.add_argument("--interactive", action="store_true", help="Запустить в интерактивном режиме")
    parser.add_argument("--lit", action="store_true", help="Запустить LIT сервер")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Используется устройство: {device}")

    tokenizer, model = load_model(args.model)
    model.to(device)
    model.eval()

    if args.interactive:
        interactive_mode(tokenizer, model, device)
    elif args.lit:
        if LIT_AVAILABLE:
            messages = [
                "Это пример сообщения для классификации.",
                "Другое тестовое сообщение для LIT.",
                "Еще одно сообщение для демонстрации работы модели."
            ]
            lit_demo, dataset = setup_lit(model, tokenizer, messages)
            if lit_demo and dataset:
                server = Server(models={'bert': lit_demo}, datasets={'example': dataset})
                server.serve()
            else:
                print("Не удалось настроить LIT. Проверьте ошибки выше.")
        else:
            print("LIT не доступен. Убедитесь, что он установлен.")
    else:
        print("Выберите режим работы: --interactive или --lit")

if __name__ == "__main__":
    username = get_username()
    print_banner(username)
    main()

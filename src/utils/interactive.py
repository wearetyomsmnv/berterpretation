from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from colorama import Fore, Back, Style as ColoramaStyle
from src.config import get_color
from src.config import valid_labels
from src.config import highlight_words
from src.config import get_colored_labels
from src.model.predict import predict_labels
from src.utils.visualization import visualize_attention, interpret_prediction_captum
from src.utils.security import check_adversarial_robustness, check_information_leakage, check_invariance, check_overfitting
import torch
import gc
import numpy
import time


def interactive_mode(tokenizer, model, device):
    style = PromptStyle.from_dict({
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
            print()
            print(f"Предсказанные метки: {' '.join(colored_labels)}")
            print()
            print("Вероятности всех меток:")
            for label, prob in zip(valid_labels, all_predictions):
                print(f"  {get_color(label)}{label}: {prob:.2%}{ColoramaStyle.RESET_ALL}")
            print(f"Важные токены для каждой метки:")
            for label, info in important_tokens.items():
                print(f"  {get_color(label)}{label}: {', '.join(info['tokens'])}{ColoramaStyle.RESET_ALL}")
            
            print("\nПроверка безопасности модели...")
            check_adversarial_robustness(model, tokenizer, device, message)
            check_information_leakage(model, tokenizer, device, ["password", "credit_card", "secret"])
            check_invariance(model, tokenizer, device, message)
    
            # Для проверки переобучения нужен тестовый набор данных
            test_sentences = {
                "This is a test sentence": 0,
                "Another test sentence": 1,
            # Добавьте больше тестовых предложений с их правильными метками
            }
            check_overfitting(model, tokenizer, device, test_sentences)
            print()
            print("\n Статистика предсказаний:")
            print()
            print(f"  Количество предсказанных меток: {len(predicted_labels)}")
            print(f"  Средняя вероятность предсказанных меток: {numpy.mean([info['score'] for info in important_tokens.values()]):.2%}")
            print(f"  Максимальная вероятность: {numpy.max(all_predictions):.2%}")
            print(f"  Минимальная вероятность: {numpy.min(all_predictions):.2%}")
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
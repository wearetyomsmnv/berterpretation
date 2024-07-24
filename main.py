import argparse
import torch
from src.model.load import load_model
from src.utils.interactive import interactive_mode
from src.lit.setup import setup_lit
from src.utils.security import check_security
from src.config import print_banner, LIT_AVAILABLE

def main():
    parser = argparse.ArgumentParser(description="Классификация текста с использованием BERT")
    parser.add_argument("--model", type=str, default="model", help="Путь к модели")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Устройство для вычислений (cuda/cpu)")
    parser.add_argument("--interactive", action="store_true", help="Запустить в интерактивном режиме")
    parser.add_argument("--lit", action="store_true", help="Запустить LIT сервер")
    parser.add_argument("--check_security", action="store_true", help="Выполнить проверки безопасности модели")
    args = parser.parse_args()

    device = torch.device(args.device)
    print()
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
                from lit_nlp.dev_server import Server
                server = Server(models={'bert': lit_demo}, datasets={'example': dataset})
                server.serve()
            else:
                print("Не удалось настроить LIT. Проверьте ошибки выше.")
        else:
            print("LIT не доступен. Убедитесь, что он установлен.")
    else:
        print("Выберите режим работы: --interactive или --lit")
    
    if args.check_security:
        check_security(model, tokenizer, device)

if __name__ == "__main__":
    print_banner()
    main()
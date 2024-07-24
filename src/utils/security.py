import torch
import torch.nn.functional as F
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import (
    TextFoolerJin2019, 
    BERTAttackLi2020, 
    PWWSRen2019, 
    DeepWordBugGao2018,
    TextBuggerLi2018
)
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding
from textattack import Attack

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def __call__(self, text_input_list):
        #print(f"Input received: {text_input_list}")  # Debug print
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]
        elif isinstance(text_input_list, tuple):
            text_input_list = list(text_input_list)
        elif not isinstance(text_input_list, list):
            raise ValueError("Input must be a string, a list of strings, or a tuple")
        
        #print(f"Processed input: {text_input_list}")  # Debug print
        inputs = self.tokenizer(text_input_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #print(f"Tokenized input: {inputs}")  # Debug print
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu().numpy()
        #print(f"Model output shape: {logits.shape}")
        #print(f"Model output: {logits}")
        return logits

    def get_grad(self, text_input):
        #print(f"get_grad input: {text_input}")  # Debug print
        if isinstance(text_input, str):
            text_input = [text_input]
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #print(f"get_grad tokenized input: {inputs}")  # Debug print
        
        self.model.zero_grad()
        outputs = self.model(**inputs)
        
        # Assuming binary classification
        labels = torch.argmax(outputs.logits, dim=1)
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()
        
        grad = inputs['input_ids'].grad.cpu().numpy()
        #print(f"get_grad output shape: {grad.shape}")  # Debug print
        #print(f"get_grad output: {grad}")  # Debug print
        return grad

def create_custom_attack(model_wrapper):
    goal_function = UntargetedClassification(model_wrapper)
    transformation = WordSwapEmbedding(max_candidates=50)
    constraints = [
        RepeatModification(),
        StopwordModification(),
        WordEmbeddingDistance(min_cos_sim=0.8)
    ]
    search_method = GreedyWordSwapWIR(wir_method="delete")

    return Attack(goal_function, constraints, transformation, search_method)

def perform_textattack_checks(model, tokenizer, device):
    print("Выполнение расширенных проверок с использованием TextAttack...")

    try:
        model_wrapper = CustomModelWrapper(model, tokenizer)
        
        # Проверка работы model_wrapper
        test_input = "This is a test sentence."
        print(f"Тестовый ввод: {test_input}")
        test_output = model_wrapper(test_input)
        #print(f"Тестовый вывод model_wrapper: {test_output}")
        
        dataset = Dataset([
            (item['text'], label) for item, label in [
                ({"text": "This is a very positive review of the product."}, 1),
                ({"text": "I absolutely hate this item, it's terrible."}, 0),
                ({"text": "The movie was okay, nothing special."}, 1),
                ({"text": "I can't recommend this enough, it's amazing!"}, 1),
                ({"text": "This is the worst experience I've ever had."}, 0),
                ({"text": "Эта ситуация неподвластна нам."}, 1),
                ({"text": "Земля круглая - мир квадратный."}, 0),
                ({"text": "Окей, этот фильм - хорош."}, 1),
                ({"text": "Мне нравится это звучание, однако я не могу сейчас послушать"}, 1),
                ({"text": "Дорога."}, 0)
            ]
        ])

        # Проверка работы model_wrapper на dataset
        for text, label in dataset:
            print(f"Текст: {text}")
            print(f"Метка: {label}")
            #print(f"Тип текста: {type(text)}")
            try:
                output = model_wrapper(text['text'])
                #print(f"Вывод модели: {output}")
            except Exception as e:
                print(f"Ошибка при обработке текста: {e}")
            print()

        attacks = [
            ("TextFooler", TextFoolerJin2019.build(model_wrapper)),
            ("BERT-Attack", BERTAttackLi2020.build(model_wrapper)),
            ("PWWS", PWWSRen2019.build(model_wrapper)),
            ("DeepWordBug", DeepWordBugGao2018.build(model_wrapper)),
            ("TextBugger", TextBuggerLi2018.build(model_wrapper))
        ]

        for attack_name, attack in attacks:
            try:
                results = []
                for example, label in dataset:
                    result = attack.attack(example, label)
                    results.append(result)

                success_rate = sum(1 for r in results if isinstance(r, SuccessfulAttackResult)) / len(results)
                print()
                print(f"Успешность атаки {attack_name}: {success_rate:.2f}")
                print()
                if success_rate > 0.5:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Модель уязвима к атаке {attack_name}")
            except Exception as e:
                print(f"Ошибка при выполнении атаки {attack_name}: {e}")
                import traceback
                #traceback.print_exc()

        # Пользовательская атака с ограничениями
        try:
            custom_attack = create_custom_attack(model_wrapper)
            custom_results = []
            for example, label in dataset:
                result = custom_attack.attack(example, label)
                custom_results.append(result)

            custom_success_rate = sum(1 for r in custom_results if isinstance(r, SuccessfulAttackResult)) / len(custom_results)
            print(f"Успешность пользовательской атаки: {custom_success_rate:.2f}")
            if custom_success_rate > 0.5:
                print("ПРЕДУПРЕЖДЕНИЕ: Модель уязвима к пользовательской атаке")
        except Exception as e:
            print(f"Ошибка при выполнении пользовательской атаки: {e}")
            import traceback
            traceback.print_exc()

        # Проверка устойчивости к небольшим изменениям
        check_robustness_to_small_changes(model_wrapper, dataset)

    except Exception as e:
        print(f"Ошибка при выполнении расширенных проверок TextAttack: {e}")
        import traceback
        traceback.print_exc()

def get_model_architecture_info(model):
    print("Получение информации об архитектуре модели...")
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Общее количество параметров: {total_params}")
        print(f"Количество обучаемых параметров: {trainable_params}")
        
        print()
        print("\nСтруктура модели:")
        print()
        print(model)
        
        print()
        print("\nРазмеры параметров по слоям:")
        print()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.size()}")
        
        # Проверка на наличие специфических слоев или структур
        has_attention = any('attention' in name for name, _ in model.named_modules())
        has_lstm = any('lstm' in name.lower() for name, _ in model.named_modules())
        has_gru = any('gru' in name.lower() for name, _ in model.named_modules())
        
        print("\nСпецифические архитектурные особенности:")
        print("_____________________________________")
        print(f"Использует механизм внимания: {'Да' if has_attention else 'Нет'}")
        print(f"Содержит LSTM слои: {'Да' if has_lstm else 'Нет'}")
        print(f"Содержит GRU слои: {'Да' if has_gru else 'Нет'}")
        
    except Exception as e:
        print(f"Ошибка при получении информации об архитектуре: {e}")
        import traceback
        traceback.print_exc()

def check_architectural_vulnerabilities(model):
    print("Проверка архитектурных уязвимостей...")
    
    try:
        # Проверка на отсутствие dropout
        has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
        if not has_dropout:
            print("ПРЕДУПРЕЖДЕНИЕ: В модели отсутствуют слои Dropout, что может привести к переобучению")
        
        # Проверка на отсутствие нормализации
        has_normalization = any(isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)) for module in model.modules())
        if not has_normalization:
            print("ПРЕДУПРЕЖДЕНИЕ: В модели отсутствуют слои нормализации, что может затруднить обучение")
        
        # Проверка на слишком глубокую архитектуру
        depth = len(list(model.modules()))
        if depth > 100:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Модель имеет большую глубину ({depth} слоев), что может привести к проблеме исчезающего градиента")
        
        # Проверка на использование активаций, подверженных проблеме исчезающего градиента
        has_sigmoid_or_tanh = any(isinstance(module, (torch.nn.Sigmoid, torch.nn.Tanh)) for module in model.modules())
        if has_sigmoid_or_tanh:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель использует Sigmoid или Tanh активации, которые могут привести к проблеме исчезающего градиента")
        
        # Проверка на отсутствие регуляризации весов
        has_weight_decay = any(hasattr(param, 'weight_decay') and param.weight_decay > 0 for param in model.parameters())
    
        if not has_weight_decay:
            print("ПРЕДУПРЕЖДЕНИЕ: В модели не используется регуляризация весов (weight decay), что может привести к переобучению")
        
        # Проверка на несбалансированность архитектуры (пример: слишком большая разница в размерах слоев)
        layer_sizes = [module.out_features for module in model.modules() if isinstance(module, torch.nn.Linear)]
        if len(layer_sizes) > 1:
            max_size_ratio = max(layer_sizes) / min(layer_sizes)
            if max_size_ratio > 10:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Обнаружена значительная несбалансированность в размерах слоев (соотношение {max_size_ratio:.2f}:1)")
        
    except Exception as e:
        print(f"Ошибка при проверке архитектурных уязвимостей: {e}")
        import traceback
        traceback.print_exc()

def check_robustness_to_small_changes(model_wrapper, dataset):
    small_changes = [
        (".", ""),
        (" ", "  "),
        ("!", "?"),
        ("a", "@"),
        ("o", "0"),
        ("i", "l"),
        ("l", "1"),
        ("н", "H"),
        ("А", "A"),
        ("с", "C"),
        ("Р", "P"),
        ("З", "3"),
        ("В", "B"),
        ("Т", "T"),
    ]

    for original, label in dataset:
        try:
            original_prediction = model_wrapper(original['text'])[0].argmax()
            for change_from, change_to in small_changes:
                modified = original['text'].replace(change_from, change_to)
                modified_prediction = model_wrapper(modified)[0].argmax()
                
                if original_prediction != modified_prediction:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Модель чувствительна к замене '{change_from}' на '{change_to}'")
                    print(f"Оригинал: {original['text']}")
                    print(f"Изменено: {modified}")
                    print(f"Оригинальное предсказание: {original_prediction}")
                    print(f"Измененное предсказание: {modified_prediction}")
                    print()
        except Exception as e:
            print(f"Ошибка при проверке устойчивости к малым изменениям: {e}")
            import traceback
            traceback.print_exc()

def check_adversarial_robustness(model, tokenizer, device, sentence, epsilon=0.01):
    try:
        # Токенизация входного предложения
        original_input = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Получение эмбеддингов для оригинального входа
        original_embeddings = model.bert.embeddings.word_embeddings(original_input['input_ids'])
        
        # Вывод информации о типах данных и размерах
        #print(f"[DEBUG] Тип input_ids: {original_input['input_ids'].dtype}")
        #print(f"[DEBUG] Размер эмбеддингов: {original_embeddings.shape}")
        
        # Получение оригинального выхода модели
        with torch.no_grad():
            original_output = model(inputs_embeds=original_embeddings, attention_mask=original_input['attention_mask']).logits
        
        # Создание возмущенных эмбеддингов
        noise = torch.randn_like(original_embeddings).to(torch.float32) * epsilon
        perturbed_embeddings = original_embeddings + noise
        
        # Получение выхода модели для возмущенных данных
        with torch.no_grad():
            perturbed_output = model(inputs_embeds=perturbed_embeddings, attention_mask=original_input['attention_mask']).logits
        
        # Вычисление разницы между оригинальным и возмущенным выходом
        difference = torch.norm(original_output - perturbed_output)
        
        print(f"Разница в выходах для состязательного примера: {difference.item()}")
        if difference > epsilon:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель может быть уязвима к состязательным примерам")
        else:
            print("Модель демонстрирует устойчивость к небольшим возмущениям")
        
        # Дополнительная информация для отладки
        #print(f"[DEBUG] Максимальное значение в оригинальном выходе: {original_output.max().item()}")
        #print(f"[DEBUG] Максимальное значение в возмущенном выходе: {perturbed_output.max().item()}")
        
    except Exception as e:
        print(f"Ошибка при проверке устойчивости к состязательным примерам: {e}")
        import traceback
        traceback.print_exc()

def check_information_leakage(model, tokenizer, device, sensitive_words):
    random_input = tokenizer("This is a random input", return_tensors="pt").to(device)
    output = model(**random_input).logits
    
    for word in sensitive_words:
        if word in tokenizer.get_vocab():
            print(f"ПРЕДУПРЕЖДЕНИЕ: Конфиденциальное слово '{word}' найдено в словаре модели")

def check_overfitting(model, tokenizer, device, test_sentences):
    correct_predictions = 0
    for sentence, label in test_sentences.items():
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        outputs = model(**inputs).logits
        predicted_class = torch.argmax(outputs, dim=1).item()
        if predicted_class == label:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_sentences)
    print(f"Точность на тестовом наборе: {accuracy:.2f}")
    if accuracy > 0.99:
        print("ПРЕДУПРЕЖДЕНИЕ: Возможно переобучение модели")

def check_invariance(model, tokenizer, device, sentence):
    original_input = tokenizer(sentence, return_tensors="pt").to(device)
    original_output = model(**original_input).logits
    
    # Добавляем пробел
    modified_sentence = sentence.replace(" ", "  ")
    modified_input = tokenizer(modified_sentence, return_tensors="pt").to(device)
    modified_output = model(**modified_input).logits
    
    difference = torch.norm(original_output - modified_output)
    print(f"Разница в выходах при добавлении пробела: {difference.item()}")
    if difference > 0.1:
        print("ПРЕДУПРЕЖДЕНИЕ: Модель может быть чувствительна к незначительным изменениям входных данных")

def check_security(model, tokenizer, device):
    print("Выполнение проверок безопасности модели...")
    
    try:
        check_adversarial_robustness(model, tokenizer, device, "This is a test sentence")
    except Exception as e:
        print(f"Ошибка при проверке устойчивости к состязательным примерам: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        check_information_leakage(model, tokenizer, device, ["password", "credit_card", "secret","фамилия","имя","отчество"])
    except Exception as e:
        print(f"Ошибка при проверке утечки информации: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        check_invariance(model, tokenizer, device, "This is a test sentence")
    except Exception as e:
        print(f"Ошибка при проверке инвариантности: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_sentences = {
            "This is a positive review": 1,
            "This is a negative review": 0,
        }
        check_overfitting(model, tokenizer, device, test_sentences)
    except Exception as e:
        print(f"Ошибка при проверке переобучения: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        get_model_architecture_info(model)
    except Exception as e:
        print(f"Ошибка при получении информации об архитектуре модели: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        check_architectural_vulnerabilities(model)
    except Exception as e:
        print(f"Ошибка при проверке архитектурных уязвимостей: {e}")
        import traceback
        traceback.print_exc()
    
    perform_textattack_checks(model, tokenizer, device)
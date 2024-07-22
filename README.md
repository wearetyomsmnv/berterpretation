<h1 align="center">berterpretation</h1>

<p align="center">
  <strong>Погрузитесь в глубины BERT с помощью мощных инструментов интерпретации.</strong>
</p>

<p align="center">
  <a href="#особенности">Особенности</a> •
  <a href="#установка">Установка</a> •
  <a href="#использование">Использование</a> •
  <a href="#требования">Требования</a> •
  <a href="#структура-проекта">Структура</a> •
  <a href="#лицензия">Лицензия</a>
</p>

---

## 🌟 Особенности

- 🧠 **Анализ BERT**: Загрузка и использование предобученных моделей BERT
- 💬 **Интерактивный режим**: Анализ текстовых сообщений в реальном времени
- 👁️ **Визуализация внимания**: Наглядное представление механизма внимания BERT
- 🔍 **Integrated Gradients**: Глубокий анализ важности токенов
- 🛠️ **Интеграция с LIT**: Расширенный анализ с помощью Language Interpretability Tool
- ! **Длина тензоров**: 512

## 🚀 Установка

1. **Клонирование репозитория**:
   ```bash
   git clone https://github.com/wearetyomsmnv/berterpretation/
   cd berterpretation
   ```

2. **Создание виртуального окружения**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # На Windows: venv\Scripts\activate
   ```

3. **Установка зависимостей**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Настройка меток и цветов для них**:

  Каждая модель умеет классифицровать по определённым меткам, их необходимо прописать в коде перед запуском скрипта:
   ```python
   valid_labels = ["about", "........ добавьте свои"]
   ```
  - в этой строке укажите свои метки

  Также, вы можете определить цвета для меток:
   ```python
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
   ```
  - меняем на свои метки, тем самым окрашивая их


## 🖥️ Использование

### Интерактивный режим

```bash
python main.py --interactive --model path/to/your/model
```

### Режим LIT

```bash
python main.py --lit --model path/to/your/model
```

## 📋 Требования

- Python 3.7+
- PyTorch
- Transformers
- Captum
- Matplotlib

<details>
  <summary>Полный список зависимостей</summary>
  
  См. файл `requirements.txt`
</details>

## 📁 Структура проекта

```
berterpretation/
│
├── main.py            # Основной скрипт
├── README.md          # Документация
└── requirements.txt   # Зависимости
```

## 📄 Лицензия

Этот проект лицензирован под [MIT License](LICENSE).

## 🤝 Вклад в проект

Мы приветствуем вклад сообщества! Если у вас есть идеи для улучшения:

1. Форкните репозиторий
2. Создайте свою ветку (`git checkout -b feature/AmazingFeature`)
3. Зафиксируйте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте изменения (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 👨‍💻 Авторы

- **Артём Семенов** - [wearetyomsmnv](https://github.com/wearetyomsmnv)

---

<p align="center">
  Создано с ❤️, wearetyomsmnv.
</p>

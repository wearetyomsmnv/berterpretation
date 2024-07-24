import pyfiglet
from colorama import init, Fore, Back, Style

init()  # Инициализация colorama для Windows

valid_labels = ["about", "aboutage", "aboutcash", "aboutcity", "aboutfriend", "abouthobby", "aboutlaw", "aboutwork", "discuss", "discusschat", "discussfraud", "discusslaw", "humor", "infotech", "infotechdevelop", "infotechhack", "infotechosint", "infotechphone", "link", "persinfo", "trash", "trashquestion"]


color_map = {
    'humor': Fore.CYAN,
    'persinfo': Fore.YELLOW,
    'about': Fore.GREEN,
    'aboutage': Fore.LIGHTGREEN_EX,
    'aboutcash': Fore.LIGHTBLUE_EX,
    'aboutcity': Fore.LIGHTCYAN_EX,
    'aboutfriend': Fore.LIGHTYELLOW_EX,
    'abouthobby': Fore.LIGHTMAGENTA_EX,
    'aboutlaw': Fore.LIGHTRED_EX,
    'aboutwork': Fore.LIGHTWHITE_EX,
    'discuss': Fore.MAGENTA,
    'discusschat': Fore.LIGHTMAGENTA_EX,
    'discussfraud': Fore.RED,
    'discusslaw': Fore.LIGHTRED_EX,
    'infotech': Fore.BLUE,
    'infotechdevelop': Fore.LIGHTBLUE_EX,
    'infotechhack': Fore.CYAN,
    'infotechosint': Fore.LIGHTCYAN_EX,
    'infotechphone': Fore.BLUE,
    'link': Fore.WHITE,
    'trash': Fore.RED,
    'trashquestion': Fore.LIGHTRED_EX
}


try:
    from lit_nlp.api import dataset as lit_dataset
    from lit_nlp.api import types as lit_types
    from lit_nlp.lib import caching
    from lit_nlp.dev_server import Server
    LIT_AVAILABLE = True
except ImportError:
    print("WARNING: LIT (Language Interpretability Tool) не может быть импортирован. Функциональность LIT будет отключена.")
    LIT_AVAILABLE = False

print()
def print_banner():
    banner_text = "berterpretation"
    
    # Генерируем ASCII-арт из текста
    ascii_art = pyfiglet.figlet_format(banner_text, font="slant")
    
    # Добавляем градиентную окраску
    colors = [Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
    colored_lines = []
    for i, line in enumerate(ascii_art.split('\n')):
        colored_line = ''.join(colors[i % len(colors)] + char for char in line)
        colored_lines.append(colored_line)
    colored_ascii = '\n'.join(colored_lines)
    
    # Создаем подпись
    signature = "bert models interpretation and security checker by @wearetyomsmnv"
    
    # Определяем ширину баннера
    width = max(len(line) for line in ascii_art.split('\n'))
    
    # Создаем рамку
    horizontal_border = Fore.YELLOW + '╔' + '═' * (width + 2) + '╗' + Fore.RESET
    bottom_border = Fore.YELLOW + '╚' + '═' * (width + 2) + '╝' + Fore.RESET
    
    # Выводим баннер
    print(horizontal_border)
    for line in colored_ascii.split('\n'):
        print(Fore.YELLOW + '║ ' + Fore.RESET + line.ljust(width) + Fore.YELLOW + ' ║' + Fore.RESET)
    
    # Добавляем подпись
    print(Fore.YELLOW + '║ ' + Fore.RESET + ' ' * ((width - len(signature)) // 2) + Fore.GREEN + signature + Fore.RESET + ' ' * ((width - len(signature) + 1) // 2) + Fore.YELLOW + ' ║' + Fore.RESET)
    
    print(bottom_border)

print()
print()

def get_color(label):
    return color_map.get(label, Fore.WHITE)

def get_colored_labels(labels, important_tokens):
    return [f"{Back.WHITE}{get_color(label)}{label} ({important_tokens.get(label, {'score': 0})['score']:.2%}){Style.RESET_ALL}" for label in labels]

def highlight_words(message, important_tokens):
    words = message.split()
    highlighted_words = []
    for word in words:
        highlighted = False
        for label, info in important_tokens.items():
            if any(token.lower() in word.lower() for token in info['tokens']):
                color = get_color(label)
                highlighted_words.append(f"{color}{word}{Style.RESET_ALL}")
                highlighted = True
                break
        if not highlighted:
            highlighted_words.append(word)
    return ' '.join(highlighted_words)
from collections import namedtuple


class Menu:
    def __init__(self):
        self._text = []

    @staticmethod
    def _press_enter_message():
        input("Нажмите Enter, чтобы продолжить...")

    def _open_file(self):
        name = self._prompt("Введите название файла")
        try:
            with open(name) as file:
                self._text = file.readlines()
        except FileNotFoundError:
            print("Файл не найден")
            self._press_enter_message()

    def _input_string(self):
        self._text.append(self._prompt("Введите строку") + '\n')

    def _show_text(self):
        for string in self._text:
            print(string, end='')
        self._press_enter_message()

    def _find_word(self):
        word = self._prompt("Введите слово")
        for i, string in enumerate(self._text):
            if word in string:
                print("Слово найдено в", i + 1, "строке:")
                print(string.replace(word, '>' + word + '<'), end='')

        self._press_enter_message()

    def _delete_word(self):
        word = self._prompt("Введите удаляемое слово")
        for i, _ in enumerate(self._text):
            self._text[i] = self._text[i].replace(word, '')

    def _clear(self):
        self._text = ""

    def _save_file(self):
        name = self._prompt("Введите название файла")
        with open(name, 'w') as file:
            file.writelines(self._text)

    _Option = namedtuple('Option', 'name action')
    _options = [
        _Option("Открыть файл", _open_file),
        _Option("Ввести строку", _input_string),
        _Option("Показать текст", _show_text),
        _Option("Поиск слова", _find_word),
        _Option("Удаление слова", _delete_word),
        _Option("Очистить терминал", _clear),
        _Option("Сохранить файл", _save_file),
    ]

    @staticmethod
    def _prompt(message):
        return input("{0}:\n> ".format(message))

    def show_menu(self):
        print("{0} Выберите действие {0}".format('=' * 12))

        for i, option in enumerate(self._options):
            print("{0}: {1}".format(i + 1, option.name))
        print()

    def handle_input(self):
        while True:
            try:
                self._options[int(self._prompt("Выберите действие")) - 1].action(self)
                break
            except (IndexError, ValueError):
                print("\nОшибка ввода, введите число от 1 до 7")


if __name__ == "__main__":
    menu = Menu()
    while True:
        menu.show_menu()
        menu.handle_input()

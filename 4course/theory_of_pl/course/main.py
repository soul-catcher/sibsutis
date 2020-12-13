__theme__ = ('Написать программу, которая будет принимать на вход произвольную контекстно-свободную грамматику и '
             'выполнит преобразование её к каноническому виду. Преобразование осуществлять поэтапно, отображая '
             'результат на каждом из этапов. Программа должна проверить построенную грамматику на эквивалентность '
             'исходной: по обеим грамматикам сгенерировать множества всех цепочек в заданном пользователем диапазоне '
             'длин и проверить их на идентичность. Для подтверждения корректности выполняемых действий предусмотреть '
             'возможность корректировки любого из построенных множеств пользователем (изменение цепочки, добавление, '
             'удаление...). При обнаружении несовпадения должна выдаваться диагностика различий – где именно '
             'несовпадения и в чём они состоят.')

__help__ = ('Формат данных файла:\n'
            'Программа принимает файлы в формате .json. Пример файла:\n'
            '''{
"VT": ["a", "b", "c"],
"VN": ["A", "B", "C", "S"],
"P": {
    "S": ["AaB", "aB", "cC"],
    "A": ["AB", "a", "b", "B"],
    "B": ["Ba", ""],
    "C": ["AB", "c"]
},
  "S": "S"
}''')

import json
import sys

from PySide2.QtCore import SIGNAL
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

import utils
from grammar import Grammar
from ui import Ui_course


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_course()
        self.ui.setupUi(self)


def update_start_sym_combobox(sym=None):
    cbox = window.ui.comboBox_stars_sym
    current = cbox.currentText()
    text = window.ui.lineEdit_non_terminals.text()
    vals = utils.split_by(text, ',')
    cbox.clear()
    cbox.addItems(vals)
    if mb_new_sym := current or sym:
        if mb_new_sym in vals:
            cbox.setCurrentText(mb_new_sym)


def print_to_actions(string: str):
    window.ui.plainTextEdit_actions.appendPlainText(string)


def get_lambda() -> str:
    sym = window.ui.lineEdit_lambda.text()
    if sym == '':
        print_to_actions('Введите символ лямбды! Он не может быть пустым')
    return sym


def fill_form_with_grammar(grammar: Grammar):
    if not (lam := get_lambda()):
        return
    ui = window.ui
    ui.lineEdit_terminals.setText(', '.join(grammar.VT))
    ui.lineEdit_non_terminals.setText(', '.join(grammar.VN))
    update_start_sym_combobox(grammar.S)
    ui.rules.clear()
    for sym, rules in grammar.P.items():
        rules = [rule if rule else lam for rule in rules]
        ui.rules.appendPlainText(f'{sym} -> {" | ".join(rules)}')


def check_grammar(grammar: Grammar) -> bool:
    if not grammar.VN:
        print_to_actions("В грамматике отсутствует список нетерминальных символов!")
        return False
    if not grammar.VT:
        print_to_actions("В грамматике отсутствует список терминальных символов!")
        return False
    if not grammar.S:
        print_to_actions("Стартовый символ не задан!")
        return False
    if intersection := set(grammar.VT) & set(grammar.VN):
        print_to_actions(f'Символы {", ".join(intersection)} находятся как во множестве терминальных, так и во '
                         f'множестве нетерминальных символов. Грамматика задана неверно!')
        return False
    if grammar.S not in grammar.VN:
        print_to_actions(f'Символ {grammar.S} не находится в списке терминальных символов. Грамматика задана неверно!')
        return False
    for sym, rules in grammar.P.items():
        if sym not in grammar.VN:
            print_to_actions(f'Символ {sym} находится в списке правил, но его нет в списке нетерминальных символов. '
                             f'Грамматика задана неверно!')
            return False
        if not rules:
            print_to_actions(f'Для символа {sym} в грамматике отсутствуют правила!')
            return False
        if sym in grammar.VT:
            print_to_actions(f'Символ {sym} находится в списке терминальных символов, но при этом используется '
                             f'в левой части правил. Грамматика задана неверно!')
            return False
        for rule in rules:
            if rules.count(rule) != 1:
                print_to_actions(f'Правило {rule} не может находиться в правой части правила более одного раза!')
                return False
            for chain_sym in rule:
                if chain_sym not in grammar.VN and chain_sym not in grammar.VT:
                    print_to_actions(f'Символ {chain_sym} не находится ни во множестве терминальных, ни во множестве '
                                     f'нетерминальных символов, но при этом есть в грамматике. '
                                     f'Грамматика задана неверно!')
                    return False
    return True


def open_grammar_file():
    filename, _ = QFileDialog.getOpenFileName(filter='Text files (*.json)')
    if filename:
        try:
            grammar = Grammar(**json.load(open(filename)))
        except (json.JSONDecodeError, TypeError):
            print_to_actions('Ошибка. Файл грамматики в неверном формате. '
                             'Откройте справку для получения информации о формате файла')
            return
        if not check_grammar(grammar):
            return
        fill_form_with_grammar(grammar)


def grammar_to_str(grammar: Grammar) -> str:
    rules = []
    for left, right in grammar.P.items():
        rules.append(f'  {left} -> {" | ".join([rule if rule else get_lambda() for rule in right])}')
    return (f'G({{{", ".join(grammar.VT)}}}, {{{", ".join(grammar.VN)}}}, P, {grammar.S})\n'
            f'P:\n' + '\n'.join(rules))


def read_grammar_from_form() -> Grammar:
    if not (lam := get_lambda()):
        return False
    ui = window.ui
    try:
        p = utils.parse_rules(ui.rules.toPlainText(), lam)
    except utils.WrongRulesException as e:
        print_to_actions(e.message)
        return False
    return Grammar(
        utils.split_by(ui.lineEdit_terminals.text(), ','),
        utils.split_by(ui.lineEdit_non_terminals.text(), ','),
        p,
        window.ui.comboBox_stars_sym.currentText()
    )

def remove_empty_and_chains_rules(grammar: Grammar) -> Grammar:
    child_free = grammar.find_child_free_non_terms()
    if child_free:
        print_to_actions(f'Обнаружены бесплодные нетерминальные символы: {", ".join(child_free)}')
        grammar = grammar.remove_rules(child_free)
        print_to_actions('Грамматика после удаления бесплодных символов:\n' + grammar_to_str(grammar))
    else:
        print_to_actions('Бесплодных нетерминальных символов не обнаружено')
    unreacheble = grammar.find_unreachable_rules()
    if unreacheble:
        print_to_actions(f'Обнаружены недостижимые нетерминальные символы: {", ".join(unreacheble)}')
        grammar = grammar.remove_rules(unreacheble)
        print_to_actions('Грамматика после удаления недостижимых символов:\n' + grammar_to_str(grammar))
    else:
        print_to_actions('Недостижимых нетерминальных символов не обнаружено')
    return grammar


def build_canon_grammar(grammar: Grammar) -> Grammar:
    print_to_actions('{:=^60}'.format('Начато построение грамматики в каноничном виде'))
    grammar = remove_empty_and_chains_rules(grammar)
    print_to_actions('Удаление пустых правил...')
    grammar = grammar.remove_empty_rules()
    print_to_actions('Грамматика после удаления пустых правил:\n' + grammar_to_str(grammar))
    print_to_actions('Удаление цепных правил...')
    grammar = grammar.remove_chain_rules()
    print_to_actions('Грамматика после удаления цепных правил:\n' + grammar_to_str(grammar))
    grammar = remove_empty_and_chains_rules(grammar)
    print_to_actions('{:=^60}'.format('Завершено построение грамматики в каноничном виде'))
    return grammar


def compare_chains():
    non_canon_chains = set(window.ui.plainTextEdit_non_canon_chains.toPlainText().splitlines())
    canon_chains = set(window.ui.plainTextEdit_canon_chains.toPlainText().splitlines())
    if non_canon_chains == canon_chains:
        window.ui.label_status.setText('Статус: множества цепочек равны')
    else:
        window.ui.label_status.setText('Статус: множества цепочек НЕ равны')
        if diff := non_canon_chains - canon_chains:
            print_to_actions('В множестве неканоничных цепочек присутствуют цепочки, '
                             'не входящие во множество каноничных цепочек:')
            print_to_actions(', '.join(diff))
        if diff := canon_chains - non_canon_chains:
            print_to_actions('В множестве каноничных цепочек присутствуют цепочки, '
                             'не входящие во множество неканоничных цепочек:')
            print_to_actions(', '.join(diff))


def calculate():
    if not (grammar := read_grammar_from_form()):
        return
    if not check_grammar(grammar):
        return
    canon_grammar = build_canon_grammar(grammar)
    window.ui.plainTextEdit_canon_grammar.setPlainText(grammar_to_str(canon_grammar))
    print_to_actions('Выполняется построение цепочек, это может занять подолжительное время..')
    QApplication.processEvents()
    sequences = grammar.make_chains(window.ui.spinBox_min_len.value(), window.ui.spinBox_max_len.value())
    sequences = [seq if seq else get_lambda() for seq in sequences]
    window.ui.plainTextEdit_non_canon_chains.setPlainText('\n'.join(sorted(sequences)))
    canon_sequences = canon_grammar.make_chains(window.ui.spinBox_min_len.value(), window.ui.spinBox_max_len.value())
    canon_sequences = [seq if seq else get_lambda() for seq in canon_sequences]
    window.ui.plainTextEdit_canon_chains.setPlainText('\n'.join(sorted(canon_sequences)))


def update_spin_box():
    window.ui.spinBox_max_len.setMinimum(window.ui.spinBox_min_len.value())

def save_result():
    filename, _ = QFileDialog.getSaveFileName(filter='Text files (*.txt)')
    if not filename:
        return
    strings = []
    strings.append('Грамматика в каноническом виде:\n')
    strings.append(window.ui.plainTextEdit_canon_grammar.toPlainText())
    strings.append('\nЦепочки, построенные из неканонической грамматики:\n')
    strings.append(window.ui.plainTextEdit_non_canon_chains.toPlainText())
    strings.append('\nЦепочки, построенные из канонической грамматики:\n')
    strings.append(window.ui.plainTextEdit_canon_chains.toPlainText())
    open(filename, 'w').writelines(strings)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()
    update_start_sym_combobox()
    window.ui.pushButton_claculate.connect(SIGNAL('clicked()'), calculate)
    window.ui.lineEdit_non_terminals.connect(SIGNAL('editingFinished()'), update_start_sym_combobox)
    window.ui.action_open.connect(SIGNAL('triggered()'), open_grammar_file)
    window.ui.action_author.connect(
        SIGNAL('triggered()'), lambda: QMessageBox.information(window, "Автор", "Самсонов Дмитрий\nГруппа: ИП-712")
    )
    window.ui.action_theme.connect(
        SIGNAL('triggered()'), lambda: QMessageBox.information(window, "Тема", __theme__)
    )
    window.ui.action_help.connect(
        SIGNAL('triggered()'), lambda: QMessageBox.information(window, "Помощь", __help__)
    )
    window.ui.action_save.connect(SIGNAL('triggered()'), save_result)
    window.ui.spinBox_min_len.connect(SIGNAL('editingFinished()'), update_spin_box)
    window.ui.pushButton_compare_chains.connect(SIGNAL('clicked()'), compare_chains)

    sys.exit(app.exec_())

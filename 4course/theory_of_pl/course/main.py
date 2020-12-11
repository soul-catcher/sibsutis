import json
import sys

from PySide2.QtCore import SIGNAL
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog

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
    vals = utils.split_by_commas(text)
    cbox.clear()
    cbox.addItems(vals)
    if mb_new_sym := current or sym:
        if mb_new_sym in vals:
            cbox.setCurrentText(mb_new_sym)


def print_to_actions(string: str):
    window.ui.plainTextEdit_actions.appendPlainText(string + '\n')


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
            for chain_sym in rule:
                if chain_sym not in grammar.VN and chain_sym not in grammar.VT:
                    print_to_actions(f'Символ {chain_sym} не находится ни во множестве терминальных, ни во множестве '
                                     f'нетерминальных символов, но при этом есть в грамматике. '
                                     f'Грамматика задана неверно!')
                    return False
    return True


def open_grammar_file():
    filename, _ = QFileDialog.getOpenFileName(filter='Text files (*.json)')
    try:
        grammar = Grammar(**json.load(open(filename)))
    except (json.JSONDecodeError, TypeError):
        print_to_actions('Ошибка. Файл грамматики в неверном формате. '
                         'Откройте справку для получения информации о формате файла')
        return
    if not check_grammar(grammar):
        return
    fill_form_with_grammar(grammar)


def read_grammar_from_form() -> Grammar:
    ui = window.ui
    return Grammar(
        utils.split_by_commas(ui.lineEdit_terminals.text()),
        utils.split_by_commas(ui.lineEdit_non_terminals.text()),

    )


def calculate():
    print("clicked")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()
    window.ui.pushButton_claculate.connect(SIGNAL('clicked()'), calculate)
    window.ui.lineEdit_non_terminals.connect(SIGNAL('editingFinished()'), update_start_sym_combobox)
    window.ui.action_open.connect(SIGNAL('triggered()'), open_grammar_file)

    sys.exit(app.exec_())

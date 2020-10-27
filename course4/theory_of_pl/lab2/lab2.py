#!/usr/bin/env python3

automate = {}
with open('automate.txt') as f:
    start = f.readline().split()[1]
    end = frozenset(f.readline().split()[1:])
    header = f.readline().split()
    non_term_inds = {}
    for i, char in enumerate(header):
        if char in non_term_inds:
            print(f'Ошибка! Cимвол {char} дублируется в алфавите')
            exit(1)
        non_term_inds[char] = i
    for line in f:
        automate[line[0]] = tuple(line[1:].split())


def launch_automate(seq) -> None:
    i = 0
    if (state := start) not in automate:
        print('Ошибка! Неверно задано начальное состояние! Оно отсутствует в таблице!')
        return
    for i, char in enumerate(seq):
        print(f'({state},{seq[i:]}) ├─ ', end='')
        if (transition := automate.get(state)) is None:
            print(f'\nОшибка! Состояние "{state}" отсутствует в таблице! Переход невозможен!')
            return
        if (idx := non_term_inds.get(char)) is None:
            print(f'\nОшибка! В цепочке встречен посторонний символ "{char}", отсутствующий в таблице!')
            return
        if (new_state := transition[idx]) == '-':
            print(f'\nОшибка: Отсутствует переход из состояния "{state}" по символу "{char}"')
            return
        state = new_state

    if state in end:
        print(f'({state},λ)')
        print('Автомат допускает цепочку', seq)
    else:
        print(f'({state},{seq[i + 1:]})')
        print('\nЦепочка кончилась, но автомат не пришёл в заключительное состояние!')


while True:
    try:
        seq = input('Введите цепочку. Для выхода из программы нажмите ^C\n> ')
    except KeyboardInterrupt:
        print()
        break
    launch_automate(seq)

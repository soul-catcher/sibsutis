#!/usr/bin/env python3

import re
import collections

Transfer = collections.namedtuple('Transfer', ['newstate', 'newstack'])
automate = collections.defaultdict(lambda: collections.defaultdict(dict))
with open('automate.txt') as f:
    start_stack = list(f.readline().split()[-1])
    start = f.readline().split()[-1]
    end = frozenset(f.readline().split()[2:])
    for line in f:
        match = re.fullmatch(r"\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\s*=\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*", line)
        state, char, stackchar, newstate, newstack = match.groups()
        newstack = list(reversed(newstack)) if newstack != '_' else []
        automate[state][stackchar][char] = Transfer(newstate, newstack)


def print_step(state, seq, stack):
    print(f'({state},{seq if seq else "ε"},{"".join(reversed(stack)) if stack else "ε"}) ├─ ', end='')


def launch_automate(seq) -> None:
    if (state := start) not in automate:
        print('Ошибка! Неверно задано начальное состояние! Оно отсутствует в таблице!')
        return
    stack = start_stack.copy()
    i = 0
    while stack:
        char = seq[i] if i < len(seq) else '_'
        print_step(state, seq[i:], stack)
        top_stack = stack.pop()
        if (transition1 := automate.get(state)) is None:
            print(f'\nОшибка! Состояние "{state}" отсутствует в таблице! Переход невозможен!')
            return
        if (transition2 := transition1.get(top_stack)) is None:
            print(f'\nОшибка: Отсутствует переход из состояния "{state}" с символом {top_stack} в стеке')
            return
        if (transition3 := transition2.get(char)) is None:
            if (transition3 := transition2.get('_')) is None:
                print(f'\nОшибка: Отсутствует переход из состояния "{state}" с символом {char} в цепочке с символом {top_stack} в стеке')
                return
            else:
                i -= 1
        state = transition3.newstate
        stack.extend(transition3.newstack)
        i += 1
    print_step(state, seq[i:], stack)
    if state in end and i >= len(seq):
        print('\nАвтомат допускает цепочку', seq)
    elif state not in end:
        print('\nСтек пуст, но автомат не пришёл в заключительное состояние!')
    else:
        print('\nСтек пуст, автомат в заключительном состоянии, но цепочка не кончилась!')


while True:
    try:
        seq = input('Введите цепочку. Для выхода из программы нажмите ^C\n> ')
    except KeyboardInterrupt:
        print()
        break
    launch_automate(seq)

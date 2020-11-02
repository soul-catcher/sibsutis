from random import random
import sys

from texttable import Texttable

table = Texttable(100)
table.header((
    "Номер партии",
    "Случайное число для игрока A",
    "Стратегия игрока А",
    "Случайное число для игрока B",
    "Стратегия игрока B",
    "Выигрыш игрока A",
    "Накопленный выигрыш A",
    "Средний выигрыш A"
))
matr = [list(map(int, line.split())) for line in open(sys.argv[1])]

p = (matr[1][1] - matr[1][0]) / (matr[1][1] - matr[0][1] + matr[0][0] - matr[1][0])
q = (matr[1][1] - matr[0][1]) / (matr[1][1] - matr[0][1] + matr[0][0] - matr[1][0])
win_total = 0
for game in range(1, 1001):
    first_player_number, second_player_number = random(), random()
    first_player_strategy = int(first_player_number < p)
    second_player_strategy = int(second_player_number < q)
    win = matr[first_player_strategy][second_player_strategy]
    win_total += win
    table.add_row((
        game,
        first_player_number,
        first_player_strategy + 1,
        second_player_number,
        second_player_strategy + 1,
        win,
        win_total,
        win_total / game
    ))
print(table.draw())

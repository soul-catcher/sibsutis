from fractions import Fraction
import itertools


def matrix_formatter(matrix):
    return '\n'.join(''.join(f"{str(item):10}" for item in row) for row in matrix)


def is_list_contains_only_zeroes(lst):
    return lst.count(0) == len(lst)


def jordan_elimination(matrix, x, y):
    matr[x] = [elem / matr[x][y] for elem in matr[x]]
    for i in range(len(matrix)):
        if i != x:
            matrix[i] = [matrix[x][y] * matrix[i][k] - matrix[x][k] * matrix[i][y] for k in range(len(matrix[0]))]


def get_base_solution(matrix):
    solution = []
    N = len(matrix)
    M = len(matrix[0])
    prev = -1
    for j in range(M - 1):
        col = [matrix[i][j] for i in range(N)]
        if col.count(0) == N - 1 and col.index(1) > prev:
            prev = col.index(1)
            solution.append(matrix[col.index(1)][-1])
        else:
            solution.append(Fraction(0))
    return solution


matr = [list(map(Fraction, line.split())) for line in open("simple_matrix.txt")]

N = len(matr)
M = len(matr[0])

z = 0
for i in range(N):
    if not is_list_contains_only_zeroes(matr[i]):
        while matr[i][i + z] == 0:
            z += 1
        jordan_elimination(matr, i, i + z)
    print(matrix_formatter(matr), end='\n\n')

matr = list(filter(lambda x: not is_list_contains_only_zeroes(x), matr))
N = len(matr)
combs = itertools.combinations(range(M - 1), N)
solutions = []
for comb in combs:
    for x, y in enumerate(comb):
        jordan_elimination(matr, x, y)
    print(matrix_formatter(matr), end='\n\n')
    solutions.append(get_base_solution(matr))


def format_solutions(solutions):
    return (f"({', '.join(str(item) for item in solution)})" for solution in solutions)


print("Базисные решения:", *format_solutions(solutions),
      "Опорные решения:", *format_solutions(filter(lambda solution: all(elem >= 0 for elem in solution), solutions)),
      sep='\n')

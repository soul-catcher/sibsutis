from fractions import Fraction
from operator import itemgetter

from texttable import Texttable


def matrix_formatter(matrix):
    table = Texttable()
    table.set_cols_dtype(["t"] * 6)
    table.add_rows(matrix, header=False)
    return table.draw()


def simplex_table_formatter(matrix, function):
    table = Texttable()
    table.set_cols_dtype(["t"] * 7)
    table.header(("б.п.", "x1", "x2", "x3", "x4", "x5", "1"))
    base = get_base(matrix)
    for base, row in zip(base, matrix):
        table.add_row((f"x{base + 1}", *row))
    table.add_row(("Z", *function))
    return table.draw()


def jordan_elimination(matrix, x, y):
    matrix[x] = [elem / matrix[x][y] for elem in matrix[x]]
    for i in range(len(matrix)):
        if i != x:
            matrix[i] = [matrix[x][y] * matrix[i][k] - matrix[x][k] * matrix[i][y] for k in range(len(matrix[0]))]


def get_ind_of_max(iterable):
    return max(enumerate(iterable), key=itemgetter(1))[0]


def get_ind_of_positive_min(iterable):
    cur_min = float('inf')
    ind_min = -1
    for i, elem in enumerate(iterable):
        if cur_min > elem >= 0:
            cur_min = elem
            ind_min = i
    return ind_min


def get_column(matrix, col):
    return (row[col] for row in matrix)


def list_add(list_a, list_b, mult):
    return (list_a[i] * mult + list_b[i] for i in range(len(list_a)))


def get_base(matrix):
    N = len(matrix)
    base = []
    for i, line in enumerate(matrix):
        for j, elem in enumerate(line):
            if elem == Fraction(1):
                col = [matrix[k][j] for k in range(N)]
                if col.count(0) == N - 1:
                    base.append(j)
    return base


def get_base_solution(matrix):
    solution = []
    N = len(matrix)
    M = len(matrix[0])

    for j in range(M - 1):
        col = [matrix[i][j] for i in range(N)]
        if col.count(0) == N - 1:
            solution.append(matrix[col.index(1)][-1])
        else:
            solution.append(Fraction(0))
    return solution


def format_solution(solution):
    return f"({', '.join(str(item) for item in solution)})"


def get_new_function(matrix, base, function):
    new_func = [0] * len(function)
    for j, base_elem in enumerate(base):
        for i, elem in enumerate(matrix[j]):
            if base_elem != i:
                new_func[i] += function[base_elem] * elem
    for i, el in enumerate(function):
        if i not in base:
            new_func[i] -= el
    new_func[-1] = - new_func[-1]
    return new_func


if __name__ == '__main__':
    with open("matr01.txt") as f:
        func = list(map(lambda x: -Fraction(x), f.readline().split()))
        base = list(map(lambda x: int(x) - 1, f.readline().split()))
        matr = [list(map(Fraction, line.split())) for line in f]

    print("Исходная функиця:", *(f"{val}x{i + 1}" for i, val in enumerate(func[:-1])), f"= {func[-1]}")

    for x, y in enumerate(base):
        print(matrix_formatter(matr))
        jordan_elimination(matr, x, y)
    print(matrix_formatter(matr))
    func = get_new_function(matr, base, func)
    print("Новая функиця:", *(f"{val}x{i + 1}" for i, val in enumerate(func[:-1])), f"= {func[-1]}")
    # simplex
    for i, elem in enumerate(func[:-1]):
        func[i] = -elem
    while True:
        print(simplex_table_formatter(matr, func))
        i_result_col = get_ind_of_max(func[:-1])
        if func[i_result_col] <= 0:
            break
        i_result_row = get_ind_of_positive_min(
            a / b if b != 0 else -1 for a, b in zip(get_column(matr, -1), get_column(matr, i_result_col)))
        matr[i_result_row] = [x / matr[i_result_row][i_result_col] for x in matr[i_result_row]]
        base[i_result_row] = i_result_col
        for i, row in enumerate(matr):
            if i == i_result_row:
                continue
            matr[i] = list(list_add(matr[i_result_row], row, -row[i_result_col]))
        func = list(list_add(matr[i_result_row], func, -func[i_result_col]))

    print("Оптимальный план:")
    print(format_solution(get_base_solution(matr)))
    print("F =", func[-1])

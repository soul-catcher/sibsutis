from fractions import Fraction

matr = [list(map(Fraction, line.split())) for line in open("hw_matr.txt")]

m = len(matr)
n = len(matr[0])


def foo(matr):
    z_c = 0
    for x in range(m):
        divider = matr[x][x + z_c]
        while divider == 0:
            z_c += 1
            divider = matr[x][x + z_c]
            if x + z_c >= m:
                return
        for col in range(x, n - z_c):
            matr[x][col + z_c] /= divider
        for line in matr:
            if line is not matr[x] and line[x + z_c] != 0:
                mult = line[x + z_c]
                for i in range(x, n):
                    line[i] -= matr[x][i] * mult
        print(*matr, sep='\n', end='\n\n')


foo(matr)

for row in matr:
    if len(set(row)) > 1 or row[0] != 0:
        for i, el in enumerate(row[:-1]):
            if el != 0:
                if el > 0:
                    print(f"+{el}x{i + 1} ", end='')
                else:
                    print(f"{el}x{i + 1} ", end='')
        print(f"= {row[-1]}")

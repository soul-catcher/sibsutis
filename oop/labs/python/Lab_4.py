import pickle


class Table:
    def __init__(self):
        self._table1 = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        ]

        self._table2 = [
            ['a', 'b', 'c', 'd', 'e'],
            ['f', 'g', 'h', 'i', 'j', 'k']
        ]

        self.arr = [self._table1, self._table2]

    def save(self):
        with open('tables.bin', 'wb') as f:
            pickle.dump(self.arr, f)

    def load(self):
        try:
            with open('tables.bin', 'rb') as f:
                self.arr = pickle.load(f)
        except FileNotFoundError:
            print("Файл не найден")

    def __str__(self):
        out = ""
        for table in self.arr:
            for row in table:
                out += row.__str__() + '\n'
            out += '\n'
        return out

    def get_element(self):
        table_num = int(input("Введите номер таблицы "))
        row_num = int(input("Введите номер строки "))
        column_num = int(input("Введите номер столбца "))
        try:
            print("Искомый элемент:" + str(self.arr[table_num - 1][row_num - 1][column_num - 1]))
        except IndexError:
            print("Элемент не найден")


t = Table()
t.save()
t_loaded = Table()
t_loaded.load()
print(t_loaded)
t_loaded.get_element()

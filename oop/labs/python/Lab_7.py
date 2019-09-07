from collections import namedtuple

from Lab_6 import Vector


class Terminal:
    def __init__(self):
        self.__products = Vector()

    def __add_product(self):
        name = self.__prompt("Введите название товара").lower()
        num = int(self.__prompt("Введите количество товара"))
        for i in self.__products:
            if i.data[0] == name:
                i.data[1] += num
                return
        self.__products.push([name, num])

    def __search(self):
        product = self.__prompt("Введите название товара")
        pos = self.__products.search(product)
        if pos == -1:
            print("Товар не найден")
        else:
            print("Товар найден на позиции " + str(pos + 1))

    def __buy(self):
        product = self.__prompt("Введите название товара").lower()
        num = int(self.__prompt("Введите количество товара"))
        pos = self.__products.search(product)
        if pos == -1:
            print("Товар не найден")
        else:
            if num < self.__products[self.__products.search(product)].data[1]:
                print("Продукт куплен")
                self.__products[self.__products.search(product)].data[1] -= num
            elif num == self.__products[self.__products.search(product)].data[1]:
                del self.__products[self.__products.search(product)]
            else:
                print("недостаточно продукта")

    def __show_products(self):
        for product in self.__products:
            print(product)

    __Option = namedtuple('Option', 'name action')
    __options = [
        __Option("Поиск товара", __search),
        __Option("Добавление товара", __add_product),
        __Option("Купить продукт", __buy),
        __Option("Вывести список продуктов", __show_products)
    ]

    @staticmethod
    def __prompt(message):
        return input("{0}:\n> ".format(message))

    def handle_input(self):
        while True:
            try:
                self.__options[int(self.__prompt("Выберите действие")) - 1].action(self)
                break
            except (IndexError, ValueError):
                print("\nОшибка ввода, введите число от 1 до", self.__options.__len__())

    def show_menu(self):
        print("{0} Выберите действие {0}".format('=' * 12))
        for i, option in enumerate(self.__options):
            print("{0}: {1}".format(i + 1, option.name))
        print()


if __name__ == "__main__":
    terminal = Terminal()
    while True:
        terminal.show_menu()
        terminal.handle_input()

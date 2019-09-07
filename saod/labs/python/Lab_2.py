from tree import Tree
import random


def show_info(tree):
    print("Размер:", tree.size())
    print("Высота:", tree.height())
    print("Средняя высота:", round(tree.medium_height(), 2))
    print("Контрольная сумма:", tree.check_sum())
    print("Обход слева направо:", tree.in_order(), '\n')


def print_table(tree_1, tree_2, tree_3):
    print("n = 100  Размер  К. сумма  Высота  Средняя высота")
    print("ИСДП    ", tree_1.size(), "   ", tree_1.check_sum(), "    ", tree_1.height(), "     ",
          tree_1.medium_height())
    print("Рекурс. ", tree_2.size(), "   ", tree_2.check_sum(), "    ", tree_2.height(), "    ", tree_2.medium_height())
    print("Дв. точ.", tree_3.size(), "   ", tree_3.check_sum(), "    ", tree_3.height(), "    ", tree_3.medium_height())


n = 100
rec_tree = Tree()
double_tree = Tree()
isdp_tree = Tree()

arr = list(range(n))
isdp_tree.isdp(arr)
random.shuffle(arr)

for i in arr:
    rec_tree.add_sdp_rec(i)
    double_tree.add_sdp_double(i)

print("ИСДП:")
show_info(isdp_tree)
print("Рекурсивное дерево поиска:")
show_info(rec_tree)
print("Дерево поиска с двойной точностью:")
show_info(double_tree)
print_table(isdp_tree, rec_tree, double_tree)

print(isdp_tree)
print(rec_tree)
print(double_tree)

import random
import threading

from tree import SDPTree, GraphicalTree

tree = SDPTree()
random.seed(0)
arr = random.sample(range(10), 10)
for i in arr:
    tree.add_sdp_rec(i)

g_tree = GraphicalTree(tree)


def deleter():
    while tree:
        print("Обход слева направо:", tree.in_order())
        del tree[int((input("Введите элемент, который следует удалить: ")))]
        g_tree.update()


threading.Thread(target=deleter).start()
g_tree.start()

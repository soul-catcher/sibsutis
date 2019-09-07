import random

from tree import AVLTree
from gtree import GraphicalTree

n = 20
random.seed(0)
arr = random.sample(range(n), n)
avl_tree = AVLTree(arr)

print("n={0} Размер  Контр. сумма  Высота  Средн.высота".format(n))
print("АВЛ   {0}     {1}          {2}       {3}".format(avl_tree.size(), avl_tree.check_sum(),
                                                        avl_tree.height(), avl_tree.medium_height()))

g_tree = GraphicalTree(avl_tree, "AVL", 1000, 400).start()

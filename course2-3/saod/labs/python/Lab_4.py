import random

from tree import AVLTree, SDPTree, GraphicalTree

avl_tree = AVLTree()
# sdp_tree = SDPTree()
random.seed(0)
# arr = random.sample(range(100), 100)
arr = range(10)
for i in arr:
    avl_tree.add(10 - i)
#     sdp_tree.add_sdp_rec(i)

g_tree = GraphicalTree(avl_tree, "Mega_tree", 1920, 1080)

print("n=100 Размер  Контр. сумма  Высота  Средн.высота")
# print("СДП  ", sdp_tree.size(), "    ", sdp_tree.check_sum(), "         ",
#       sdp_tree.height(), "     ", sdp_tree.medium_height())
print("АВЛ  ", avl_tree.size(), "    ", avl_tree.check_sum(), "         ",
      avl_tree.height(), "     ", avl_tree.medium_height())

g_tree.start()

import math
import tkinter as tk

import binarytree


class _Vertex:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class Tree:
    def __init__(self):
        self._root = None

    @property
    def root(self):
        return self._root

    def __str__(self):
        def __str__(root, tree):
            if root:
                tree = binarytree.Node(root.data)
                tree.left = __str__(root.left, tree.left)
                tree.right = __str__(root.right, tree.right)
            return tree

        return str(__str__(self._root, None))

    def in_order(self):
        def in_order(root):
            array = []
            if root:
                array += in_order(root.left)
                array.append(root.data)
                array += in_order(root.right)
            return array

        return in_order(self._root)

    def size(self):
        def size(root):
            return 1 + size(root.left) + size(root.right) if root else 0

        return size(self._root)

    def height(self):
        def height(root):
            return 1 + max(height(root.left), height(root.right)) if root else 0

        return height(self._root)

    def __len_sum(self, root, s):
        return s + self.__len_sum(root.left, s + 1) + self.__len_sum(root.right, s + 1) if root else 0

    def medium_height(self):
        return self.__len_sum(self._root, 1) / self.size()

    def check_sum(self):
        def check_sum(root):
            return root.data + check_sum(root.left) + check_sum(root.right) if root else 0

        return check_sum(self._root)

    def __search(self, data):
        p = self._root
        while p:
            if data < p.data:
                p = p.left
            elif data > p.data:
                p = p.right
            else:
                break
        return p

    def __bool__(self):
        return bool(self._root)


class ISDPTree(Tree):
    def build(self, array):
        def build(l, r):
            if l > r:
                return None
            else:
                m = math.ceil((l + r) / 2)
                return _Vertex(array[m], build(l, m - 1), build(m + 1, r))

        self._root = build(0, len(array) - 1)


class SDPTree(Tree):
    def add_sdp_rec(self, data):
        def add_sdp_rec(root):
            if root is None:
                return _Vertex(data)
            elif data < root.data:
                root.left = add_sdp_rec(root.left)
            elif data > root.data:
                root.right = add_sdp_rec(root.right)
            else:
                print("Данные уже в дереве")
            return root

        self._root = add_sdp_rec(self._root)

    def add_sdp_double(self, data):
        if self._root is None:
            self._root = _Vertex(data)
            return

        p = self._root
        while True:
            if data < p.data:
                if p.left:
                    p = p.left
                else:
                    p.left = _Vertex(data)
                    return
            elif data > p.data:
                if p.right:
                    p = p.right
                else:
                    p.right = _Vertex(data)
                    return
            else:
                print("Данные уже в дереве")
                break

    def __delitem__(self, item):
        p = self._root
        parent = _Vertex(-1, right=p)

        # Поиск элемента
        while p:
            if item < p.data:
                parent = p
                p = p.left
            elif item > p.data:
                parent = p
                p = p.right
            else:
                break

        if p:
            if p.left is None:
                if item < parent.data:
                    parent.left = p.right
                else:
                    parent.right = p.right
            elif p.right is None:
                if item < parent.data:
                    parent.left = p.left
                else:
                    parent.right = p.left
            else:
                r = p.left
                s = p
                if r.right is None:
                    r.right = p.right
                    if item < parent.data:
                        parent.left = r
                    else:
                        parent.right = r
                else:
                    while r.right:
                        s = r
                        r = r.right
                    s.right = r.left
                    r.left = p.left
                    r.right = p.right
                    if item < parent.data:
                        parent.left = r
                    else:
                        parent.right = r
        if parent.data == -1:
            self._root = parent.right


class _AVLVertex(_Vertex):
    def __init__(self, data, left=None, right=None):
        super().__init__(data, left, right)
        self.balance = 0


class AVLTree(Tree):

    def __init__(self, arr=None):
        super().__init__()
        self.dec = False
        if arr:
            for element in arr:
                self.add(element)

    @staticmethod
    def _ll(p):
        q = p.left
        p.balance = q.balance = 0
        p.left = q.right
        q.right = p
        return q

    def _ll1(self, p):
        q = p.left
        if q.balance == 0:
            q.balance = 1
            p.balance = -1
            self._dec = False
        else:
            q.balance = p.balance = 0
        p.left = q.right
        q.right = p
        return q

    @staticmethod
    def _rr(p):
        q = p.right
        p.balance = q.balance = 0
        p.right = q.left
        q.left = p
        return q

    def _rr1(self, p):
        q = p.right
        if q.balance == 0:
            q.balance = -1
            p.balance = 1
            self._dec = False
        else:
            q.balance = p.balance = 0
        p.right = q.left
        q.left = p
        return q

    @staticmethod
    def _lr(p):
        q = p.left
        r = q.right
        if r.balance < 0:
            p.balance = 1
        else:
            p.balance = 0
        if r.balance > 0:
            q.balance = -1
        else:
            q.balance = 0
        r.balance = 0
        q.right = r.left
        p.left = r.right
        r.left = q
        r.right = p
        return r

    @staticmethod
    def _rl(p):
        q = p.right
        r = q.left
        if r.balance > 0:
            p.balance = -1
        else:
            p.balance = 0
        if r.balance < 0:
            q.balance = 1
        else:
            q.balance = 0
        r.balance = 0
        q.left = r.right
        p.right = r.left
        r.right = q
        r.left = p
        return r

    def add(self, data):
        def add(root):
            if root is None:
                self.grow = True
                return _AVLVertex(data)
            elif root.data > data:
                root.left = add(root.left)
                if self.grow:
                    if root.balance > 0:
                        root.balance = 0
                        self.grow = False
                    elif root.balance == 0:
                        root.balance = -1
                        self.grow = True
                    else:
                        if root.left.balance < 0:
                            root = self._ll(root)
                            self.grow = False
                        else:
                            root = self._lr(root)
                            self.grow = False
            elif root.data < data:
                root.right = add(root.right)
                if self.grow:
                    if root.balance < 0:
                        root.balance = 0
                        self.grow = False
                    elif root.balance == 0:
                        root.balance = 1
                        self.grow = True
                    else:
                        if root.right.balance > 0:
                            root = self._rr(root)
                            self.grow = False
                        else:
                            root = self._rl(root)
                            self.grow = False
            else:
                print("Данные уже в дереве")
            return root

        self._root = add(self._root)

    def _bl(self, p):
        if p.balance == -1:
            p.balance = 0
            return p
        elif p.balance == 0:
            p.balance = 1
            self._dec = False
            return p
        elif p.balance == 1:
            if p.right.balance >= 0:
                return self._rr1(p)
            else:
                return self._rl(p)

    def _br(self, p):
        if p.balance == 1:
            p.balance = 0
            return p
        elif p.balance == 0:
            p.balance = -1
            self._dec = False
            return p
        elif p.balance == -1:
            if p.left.balance <= 0:
                return self._ll1(p)
            else:
                return self._lr(p)

    def __delitem__(self, x):
        def __delitem__(p, parent):
            if p is None:
                print("Вершины нет")
            elif x < p.data:
                p.left = __delitem__(p.left, p)
                if self._dec:
                    p = self._bl(p)
            elif x > p.data:
                p.right = __delitem__(p.right, p)
                if self._dec:
                    p = self._br(p)
            else:
                q = p
                if q.left is None:
                    p = q.right
                    self._dec = True
                elif q.right is None:
                    p = q.left
                    self._dec = True
                else:
                    if x > parent.data:
                        parent.right = self._rm_vertex(q.left, q)
                    elif x < parent.data:
                        parent.left = self._rm_vertex(q.left, q)
                    else:
                        tmp = self._rm_vertex(q.left, q)
                        if self._root is tmp:
                            self._root = tmp
                        else:
                            parent.left = tmp

                    if self._dec:
                        p = self._bl(p)
            return p
        self._root = __delitem__(self._root, self._root)

    def _rm_vertex(self, p, q, parent=None):
        if p.right:
            self._rm_vertex(p.right, q, p)
            if self._dec:
                return self._br(p)
        else:
            q.data = p.data
            self._dec = True
            if parent:
                parent.right = None
            else:
                q.left = p.left
            return q


class GraphicalTree:
    def __init__(self, tree, title=None, width=800, height=600):
        self._tree = tree
        self._window = tk.Tk()
        self._window.title(title)
        self._canvas = tk.Canvas(self._window, width=width, height=height)
        self._canvas.pack()
        self._canvas.bind("<Button-4>", self._scroll)
        self._canvas.bind("<Button-5>", self._scroll)
        self._canvas.bind("<ButtonPress-1>", self._drag_start)
        self._canvas.bind("<B1-Motion>", self._drag_move)

    def _scroll(self, event):
        if event.num == 4 or event.delta == 120:
            self._canvas.scale("all", event.x, event.y, 1.1, 1.1)
        elif event.num == 5 or event.delta == -120:
            self._canvas.scale("all", event.x, event.y, 0.9, 0.9)

    def _drag_start(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _drag_move(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)
        self._canvas.update()

    def _draw_vertex(self, data, x, y):
        radius = 20
        self._canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white")
        self._canvas.create_text(x, y, text=data, font=(None, round(20)))

    def _draw_tree(self, vertex, level=1):
        cell_size = 40
        if vertex:
            if vertex.left:
                self._canvas.create_line((vertex.data + 1) * cell_size, level * cell_size,
                                         (vertex.left.data + 1) * cell_size, (level + 1) * cell_size)
            if vertex.right:
                self._canvas.create_line((vertex.data + 1) * cell_size, level * cell_size,
                                         (vertex.right.data + 1) * cell_size, (level + 1) * cell_size)

            self._draw_vertex(vertex.data, (vertex.data + 1) * cell_size, level * cell_size)
            self._draw_tree(vertex.left, level + 1)
            self._draw_tree(vertex.right, level + 1)

    def start(self):
        self.update()
        self._window.mainloop()

    def update(self):
        self._canvas.delete("all")
        self._draw_tree(self._tree.root)
        self._canvas.update()

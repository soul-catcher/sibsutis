from abc import ABC, abstractmethod


class _Vertex:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class _AvlVertex(_Vertex):
    def __init__(self, data, balance=0, left=None, right=None):
        super().__init__(data, left, right)
        self.balance = balance


class Tree(ABC):
    def __init__(self):
        self._root = None

    @property
    def root(self):
        return self._root

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

    def __str__(self):
        return str(self.in_order())

    @abstractmethod
    def add(self, data):
        pass

    @abstractmethod
    def pop(self, x):
        pass


class AVLTree(Tree):
    def __init__(self, arr=None):
        super().__init__()
        self.__dec = False
        if arr:
            for element in arr:
                self.add(element)

    @staticmethod
    def __ll(p):
        q = p.left
        p.balance = q.balance = 0
        p.left = q.right
        q.right = p
        return q

    def __ll1(self, p):
        q = p.left
        if q.balance == 0:
            q.balance = 1
            p.balance = -1
            self.__dec = False
        else:
            q.balance = p.balance = 0
        p.left = q.right
        q.right = p
        return q

    @staticmethod
    def __rr(p):
        q = p.right
        p.balance = q.balance = 0
        p.right = q.left
        q.left = p
        return q

    def __rr1(self, p):
        q = p.right
        if q.balance == 0:
            q.balance = -1
            p.balance = 1
            self.__dec = False
        else:
            q.balance = p.balance = 0
        p.right = q.left
        q.left = p
        return q

    @staticmethod
    def __lr(p):
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
    def __rl(p):
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
                self.__grow = True
                return _AvlVertex(data)
            elif root.data > data:
                root.left = add(root.left)
                if self.__grow:
                    if root.balance > 0:
                        root.balance = 0
                        self.__grow = False
                    elif root.balance == 0:
                        root.balance = -1
                        self.__grow = True
                    else:
                        if root.left.balance < 0:
                            root = self.__ll(root)
                            self.__grow = False
                        else:
                            root = self.__lr(root)
                            self.__grow = False
            elif root.data < data:
                root.right = add(root.right)
                if self.__grow:
                    if root.balance < 0:
                        root.balance = 0
                        self.__grow = False
                    elif root.balance == 0:
                        root.balance = 1
                        self.__grow = True
                    else:
                        if root.right.balance > 0:
                            root = self.__rr(root)
                            self.__grow = False
                        else:
                            root = self.__rl(root)
                            self.__grow = False
            else:
                print("Данные уже в дереве")
            return root

        self._root = add(self._root)

    def __bl(self, p):
        if p.balance == -1:
            p.balance = 0
            return p
        elif p.balance == 0:
            p.balance = 1
            self.__dec = False
            return p
        elif p.balance == 1:
            if p.right.balance >= 0:
                return self.__rr1(p)
            else:
                return self.__rl(p)

    def __br(self, p):
        if p.balance == 1:
            p.balance = 0
            return p
        elif p.balance == 0:
            p.balance = -1
            self.__dec = False
            return p
        elif p.balance == -1:
            if p.left.balance <= 0:
                return self.__ll1(p)
            else:
                return self.__lr(p)

    def pop(self, x):
        def pop(p, parent):
            if p is None:
                print("Вершины нет")
            elif x < p.data:
                p.left = pop(p.left, p)
                if self.__dec:
                    p = self.__bl(p)
            elif x > p.data:
                p.right = pop(p.right, p)
                if self.__dec:
                    p = self.__br(p)
            else:
                q = p
                if q.left is None:
                    p = q.right
                    self.__dec = True
                elif q.right is None:
                    p = q.left
                    self.__dec = True
                else:
                    if x > parent.data:
                        parent.right = self.__rm_vertex(q.left, q)
                    elif x < parent.data:
                        parent.left = self.__rm_vertex(q.left, q)
                    else:
                        tmp = self.__rm_vertex(q.left, q)
                        if self._root is tmp:
                            self._root = tmp
                        else:
                            parent.left = tmp

                    if self.__dec:
                        p = self.__bl(p)
            return p

        self._root = pop(self._root, self._root)

    def __rm_vertex(self, p, q, parent=None):
        if p.right:
            self.__rm_vertex(p.right, q, p)
            if self.__dec:
                return self.__br(p)
        else:
            q.data = p.data
            self.__dec = True
            if parent:
                parent.right = None
            else:
                q.left = p.left
            return q

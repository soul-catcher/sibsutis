from abc import ABC, abstractmethod


class _Node:
    def __init__(self, data, next_=None):
        self.data = data
        self.next = next_


class _List(ABC):
    def __init__(self):
        self._list = None

    @abstractmethod
    def push(self, data):
        pass

    @abstractmethod
    def pop(self):
        pass


class Stack(_List):
    def push(self, data):
        if self._list is None:
            self._list = _Node(data)
        else:
            self._list = _Node(data, self._list)

    def pop(self):
        temp = self._list.data
        self._list = self._list.next
        return temp


class Queue(_List):
    def __init__(self):
        super().__init__()
        self._tail = None

    def push(self, data):
        if self._list is None:
            self._list = self._tail = _Node(data)
        else:
            self._tail.next = _Node(data)
            self._tail = self._tail.next

    def pop(self):
        temp = self._list.data
        self._list = self._list.next
        return temp


if __name__ == "__main__":
    q = Queue()
    s = Stack()
    for i in range(10):
        q.push(i)
        s.push(i)

    print("Stack:")
    for _ in range(10):
        print(s.pop(), end=' ')
    print("\nQueue:")
    for _ in range(10):
        print(q.pop(), end=' ')
    print()

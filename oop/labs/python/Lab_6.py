from Lab_5 import _List


class _Node:
    def __init__(self, data, previous=None, next_=None):
        self.data = data
        self.previous = previous
        self.next = next_

    def __str__(self):
        return str(self.data)


class Vector(_List):
    def __init__(self):
        super().__init__()
        self.__length = 0

    def __str__(self):
        iter_ = self._list
        arr = ""
        while iter_:
            arr += str(iter_.data) + ' '
            iter_ = iter_.next
        return arr

    def __iter__(self):
        self.iter = self._list
        return self

    def __next__(self):
        if self.iter:
            x = self.iter
            self.iter = self.iter.next
            return x
        else:
            raise StopIteration

    def __len__(self):
        return self.__length

    def push(self, data, destination=0):
        if self._list is None:
            self._list = _Node(data)
        elif destination == 0:
            self._list = _Node(data, next_=self._list)
            self._list.next.previous = self._list
        else:
            p = self._list
            for _ in range(destination - 1):
                try:
                    p = p.next
                except AttributeError:
                    print("Выход за границы массива")
                    return None
            p.next = _Node(data, p, p.next)
            p.next.previous = p
        self.__length += 1

    def pop(self):
        temp = self._list.data
        self._list = self._list.next
        self._list.previous = None
        self.__length -= 1
        return temp

    def search(self, element):
        count = 0
        p = self._list
        while p:
            if p.data[0] == element:
                return count
            count += 1
            p = p.next
        return -1

    def __getitem__(self, item):
        current = self._list
        iter_ = 0
        while iter_ < item:
            try:
                current = current.next
            except AttributeError:
                print("Выход за границы массива")
                return None
            iter_ += 1
        return current

    def __setitem__(self, key, value):
        try:
            self.__getitem__(key).data = value
        except AttributeError:
            pass

    def __delitem__(self, key):
        item = self.__getitem__(key)
        if key == 0:
            self._list = self._list.next
            if len(self) > 1:
                self._list.previous = None
        elif key == len(self):
            item.previous.next = None
        else:
            item.previous.next = item.next
            item.next.previous = item.previous
            self.__length -= 1


if __name__ == "__main__":
    v = Vector()
    v.push("one")
    v.push("two")
    v.push("three")
    v.push("four")
    print(v)
    v.push(5, 4)
    print(v)
    v[2] = 6
    print(v)
    del v[2]
    print(v)
    v.search("one")

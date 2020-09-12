#include <random>
#include <iostream>
#include <cstdlib>

class List {
    struct Node {
        Node(int data, Node *pNext) : data(data), pNext(pNext) {}
        Node *pNext;
        int data;
    } *head = nullptr, *tail = nullptr;

public:

    ~List() {
        clear();
    }

    void add(const int data) {
        head = new Node(data, head);
    }

    void addAt(const int pos, const int data) {
        if (pos == 0) {
            add(pos);
        } else {
            Node *p = head;
            for (int i = 0; i < pos; i++, p = p->pNext) {}
            p->pNext = new Node(data, p->pNext);
        }
    }

    void add_to_tail(const int data) {
        if (!head) {
            head = tail = new Node(data, nullptr);
        } else {
            tail->pNext = new Node(data, nullptr);
            tail = tail->pNext;
        }
    }

    void pop() {
        Node *temp = head;
        head = head->pNext;
        delete temp;
    }

    void delAt(const int position) {
        if (position == 0) {
            pop();
        } else {
            Node *p = head;
            for (int i = 0; i < position - 1; i++, p = p->pNext) {}
            Node *temp = p->pNext->pNext;
            delete p->pNext;
            p->pNext = temp;
        }
    }

    int size() const {
        int counter = 0;
        Node *p = head;
        while (p) {
            p = p->pNext;
            counter++;
        }
        return counter;
    }

    void clear() {
        while (head) {
            pop();
        }
    }

    void move(const int from, const int to) {
        Node *obj;
        if (from == 0) {
            obj = head;
            head = head->pNext;
        } else {
            Node *p = head;
            for (int i = 0; i < from - 1; i++, p = p->pNext) {}
            obj = p->pNext;
            p->pNext = p->pNext->pNext;
        }
        if (to == 0) {
            obj->pNext = head;
            head = obj;
        } else {
            Node *p = head;
            for (int i = 0; i < to - 1; i++, p = p->pNext) {}
            obj->pNext = p->pNext;
            p->pNext = obj;
        }
    }

    Node *operator[](int n) {
        Node *p = head;
        for (int i = 0; i < n; ++i, p = p->pNext) {}
        return p;
    }

private:
    friend std::ostream &operator<<(std::ostream &out, List &list);

    friend std::ostream &operator<<(std::ostream &out, Node *node);
};

std::ostream &operator<<(std::ostream &out, List &list) {
    for (List::Node *iter = list.head; iter; iter = iter->pNext) {
        out << iter << ' ';
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, List::Node *node) {
    out << node->data;
    return out;
}

int main() {
    const int n = 10;
    int arr[n];
    List list;
    std::cout << "Stack :\n";
    for (int &i : arr) {
        i = rand() % 5;
        list.add(i * i);
        std::cout << list << '\n';
    }
    std::cout << list << std::endl;
    int listSize = n;
    for (int i = 0; i < listSize; i++) {
        for (int j = i + 1; j < listSize; j++) {
            if (list[i]->data == list[j]->data) {
                list.delAt(j--);
                listSize--;
            }
        }
    }
    std::cout << list
              << "\nLenght = " << list.size() << '\n';

    std::cout << "Queue\n";
    List queue;
    for (int &i : arr) {
        i = rand() % 5;
        queue.add_to_tail(i);
        std::cout << queue << '\n';
    }
}

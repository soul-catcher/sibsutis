#pragma once

#include <iostream>
#include <cmath>
#include <functional>

struct Vertex {
    const int data;
    Vertex *left, *right;
    int balance;

    explicit Vertex(int data, Vertex *left = nullptr, Vertex *right = nullptr, int balance = 0) :
            data(data), left(left), right(right), balance(balance) {}
};

class Tree {
protected:
    Vertex *root = nullptr;

public:

    ~Tree() {
        std::function<void(Vertex *)> del = [&del](Vertex *root) {
            if (root) {
                del(root->right);
                del(root->left);
                delete root;
            }
        };
        del(root);
    }

    Vertex *getRoot() const {
        return root;
    }

    int size() const {
        std::function<int(Vertex *)> size = [&size](Vertex *root) {
            return root ? 1 + size(root->left) + size(root->right) : 0;
        };
        return size(root);
    }

    int height() const {
        std::function<int(Vertex *)> height = [&height](Vertex *root) {
            return root ? 1 + std::max(height(root->left), height(root->right)) : 0;
        };
        return height(root);
    }

    int checkSum() const {
        std::function<int(Vertex *)> checkSum = [&checkSum](Vertex *root) {
            return root ? root->data + checkSum(root->left) + checkSum(root->right) : 0;
        };
        return checkSum(root);
    }

private:
    int lenSum(Vertex *root, int depth) const {
        return root ? depth + lenSum(root->left, depth + 1) + lenSum(root->right, depth + 1) : 0;
    }

public:
    double mediumHeight() const {
        return (double) lenSum(root, 1) / size();
    }

    void printLNR() const {
        std::function<void(Vertex *)> printLNR = [&printLNR](Vertex *root) {
            if (root) {
                printLNR(root->left);
                std::cout << root->data << ' ';
                printLNR(root->right);
            }
        };
        return printLNR(root);
    }
};


class IsdpTree : public Tree {
public:
    IsdpTree(int arr[], int n) {
        std::function<Vertex *(int[], int, int)> IsdpTree = [&IsdpTree](int arr[], int l, int r) -> Vertex * {
            if (l > r) {
                return nullptr;
            } else {
                const int m = (int) ceil((double) (l + r) / 2);
                return new Vertex(arr[m], IsdpTree(arr, l, m - 1), IsdpTree(arr, m + 1, r));
            }
        };
        root = IsdpTree(arr, 0, n - 1);
    }
};

class SdpTree : public Tree {
public:
    void addRec(int data) {
        std::function<void(Vertex *&)> addRec = [&addRec, &data](Vertex *&root) {
            if (!root) {
                root = new Vertex(data);
            } else if (data < root->data) {
                addRec(root->left);
            } else if (data > root->data) {
                addRec(root->right);
            } else {
                std::cout << "Data already exist\n";
            }
        };
        return addRec(root);
    }

    void addDouble(int data) {
        auto p = &root;
        while (*p) {
            if (data < (*p)->data) {
                p = &((*p)->left);
            } else if (data > (*p)->data) {
                p = &((*p)->right);
            } else {
                std::cout << "Data already exist\n";
                break;
            }
        }
        if (!*p) {
            (*p) = new Vertex(data);
        }
    }
};

class AvlTree : public Tree {
    bool grow;

    static void ll(Vertex *&p) {
        Vertex *q = p->left;
        p->balance = q->balance = 0;
        p->left = q->right;
        q->right = p;
        p = q;
    }

    static void rr(Vertex *&p) {
        Vertex *q = p->right;
        p->balance = q->balance = 0;
        p->right = q->left;
        q->left = p;
        p = q;
    }

    static void lr(Vertex *&p) {
        Vertex *q = p->left;
        Vertex *r = q->right;
        if (r->balance < 0) {
            p->balance = 1;
        } else {
            p->balance = 0;
        }
        if (r->balance > 0) {
            q->balance = -1;
        } else {
            q->balance = 0;
        }
        r->balance = 0;
        q->right = r->left;
        p->left = r->right;
        r->left = q;
        r->right = p;
        p = r;
    }

    static void rl(Vertex *&p) {
        Vertex *q = p->right;
        Vertex *r = q->left;
        if (r->balance > 0) {
            p->balance = -1;
        } else {
            p->balance = 0;
        }
        if (r->balance < 0) {
            q->balance = 1;
        } else {
            q->balance = 0;
        }
        r->balance = 0;
        q->left = r->right;
        p->right = r->left;
        r->right = q;
        r->left = p;
        p = r;
    }

public:
    void add(int data) {
        std::function<void(Vertex *&)> add = [&](Vertex *&p) {
            if (!p) {
                p = new Vertex(data);
                grow = true;
            } else if (p->data > data) {
                add(p->left);
                if (grow) {
                    if (p->balance > 0) {
                        p->balance = 0;
                        grow = false;
                    } else if (p->balance == 0) {
                        p->balance = -1;
                        grow = true;
                    } else {
                        if (p->left->balance < 0) {
                            ll(p);
                            grow = false;
                        } else {
                            lr(p);
                            grow = false;
                        }
                    }
                }
            } else if (p->data < data) {
                add(p->right);
                if (grow) {
                    if (p->balance < 0) {
                        p->balance = 0;
                        grow = false;
                    } else if (p->balance == 0) {
                        p->balance = 1;
                        grow = true;
                    } else {
                        if (p->right->balance > 0) {
                            rr(p);
                            grow = false;
                        } else {
                            rl(p);
                            grow = false;
                        }
                    }
                }
            } else {
                std::cout << "Data already exist";
            }
        };
        return add(root);
    }
};

class DbdTree : public Tree {
    int vr = 1;
    int hr = 1;

public:
    void add(int data) {
        std::function<void(Vertex *&)> add = [&](Vertex *&p) {
            if (!p) {
                p = new Vertex(data);
                vr = 1;
            } else if (p->data > data) {
                add(p->left);
                if (vr == 1) {
                    if (p->balance == 0) {
                        Vertex *q = p->left;
                        p->left = q->right;
                        q->right = p;
                        p = q;
                        q->balance = 1;
                        vr = 0;
                        hr = 1;
                    } else {
                        p->balance = 0;
                        vr = 1;
                        hr = 0;
                    }
                } else {
                    hr = 0;
                }
            } else if (p->data < data) {
                add(p->right);
                if (vr == 1) {
                    p->balance = 1;
                    hr = 1;
                    vr = 0;
                } else if (hr == 1) {
                    if (p->balance == 1) {
                        Vertex *q = p->right;
                        p->balance = 0;
                        q->balance = 0;
                        p->right = q->left;
                        q->left = p;
                        p = q;
                        vr = 1;
                        hr = 0;
                    } else {
                        hr = 0;
                    }
                }
            }
        };
        return add(root);
    }

    int levels() const {
        std::function<int(Vertex *)> height = [&height](Vertex *root) {
            return root ? 1 - root->balance + std::max(height(root->left), height(root->right)) : 0;
        };
        return height(root);
    }
};

// class DopTree : public SdpTree {
//     static const int N = 10;
//     int A[N];
//     int Aw[N][N]; //матрица весов
//     int Ap[N][N];
//     int Ar[N][N];
// public:
//     int w[N];
// private:
//     int V[N];
//     int h[N];
//     void AW() {
//         for (int i = 0; i < N; i++) {
//             for (int j = i + 1; j < N; j++) {
//                 Aw[i][j] = Aw[i][j - 1] + w[j];
//             }
//         }
//     }

//     void AP_AR() {
//         int i, j, m, min, k;
//         for (i = 0; i < N - 1; i++) {
//             j = i + 1;
//             Ap[i][j] = Aw[i][j];
//             Ar[i][j] = j;
//         }
//         for (int h = 2; h < N; h++) {
//             for (i = 0; i < N - h; i++) {
//                 j = i + h;
//                 m = Ar[i][j - 1];
//                 min = Ap[i][m - 1] + Ap[m][j];
//                 for (k = m + 1; k < Ar[i + 1][j]; k++) {
//                     int x = Ap[i][k - 1] + Ap[k][j];
//                     if (x < min) {
//                         m = k;
//                         min = x;
//                     }
//                 }
//                 Ap[i][j] = min + Aw[i][j];
//                 Ar[i][j] = m;
//             }
//         }
//     }

//     void FillInc() {
//         for (int i = 0; i < N; i++) {
//             V[i] = rand() % N;
//             w[i] = rand() % 100;
//             int j = 0;
//             while (j < i) {
//                 if (V[i] == V[j]) {
//                     V[i] = rand() % N;
//                     j = -1;
//                 }
//                 j++;
//             }
//         }
//     }

//     void Create_Tree(int l, int r) { // L,R - границы массива вершин V
//         if (l < r) {
//             int k = Ar[l][r];
//             addRec(V[k]);
//             Create_Tree(l, k - 1);
//             Create_Tree(k, r);
//         }
//     }

// public:

//     void PrintMas() {
//         for (int i : V) {
//             std::cout << i << " ";
//         }
//         std::cout << '\n';
//         for (int i : w) {
//             std::cout << i << " ";
//         }
//         std::cout << '\n';
//     }

//     DopTree() {
//         FillInc();
//         AW();
//         AP_AR();
//         Create_Tree(0, N - 1);
//     }
// };
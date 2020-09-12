#include <iostream>
#include <cmath>

struct Vertex {
    int data;
    Vertex *left, *right;
};

void printLNR(Vertex *root) {
    if (root) {
        printLNR(root->left);
        std::cout << root->data << ' ';
        printLNR(root->right);
    }
}

int treeSize(Vertex *root) {
    if (root) {
        return 1 + treeSize(root->left) + treeSize(root->right);
    } else {
        return 0;
    }
}

int treeHeight(Vertex *root) {
    if (!root) {
        return 0;
    } else {
        return 1 + std::max(treeHeight(root->left), treeHeight(root->right));
    }
}

int treeLenSum(Vertex *root, int l) {
    if (!root) {
        return 0;
    } else {
        return l + treeLenSum(root->left, l + 1) + treeLenSum(root->right, l + 1);
    }
}

double treeMediumHeight(Vertex *root) {
    return (double) treeLenSum(root, 1) / treeSize(root);
}

Vertex *addVertex(int data, Vertex *left, Vertex *right) {
    auto *vertex = new Vertex;
    vertex->data = data;
    vertex->left = left;
    vertex->right = right;
    return vertex;
}

int checkSum(Vertex *root) {
    if (root) {
        return root->data + checkSum(root->left) + checkSum(root->right);
    } else {
        return 0;
    }
}

void delTree(Vertex *root) {
    if (root) {
        delTree(root->right);
        delTree(root->left);
        delete root;
    }
}

void showInfo(Vertex *root) {
    std::cout << "Size: " << treeSize(root)
              << "\nHeight: " << treeHeight(root)
              << "\nMedium height: " << treeMediumHeight(root)
              << "\nCheck sum: " << checkSum(root)
              << "\nIn-order traversal (LNR): ";
    printLNR(root);
    std::cout << std::endl;
}

struct Trunk {
    Trunk *prev;
    std::string str;

    Trunk(Trunk *prev, std::string str) {
        this->prev = prev;
        this->str = std::move(str);
    }
};

void showTrunks(Trunk *p) {
    if (p == nullptr)
        return;

    showTrunks(p->prev);

    std::cout << p->str;
}

void printTree(Vertex *root, Trunk *prev = nullptr, bool isLeft = false) {
    if (root == nullptr)
        return;

    std::string prev_str = "    ";
    Trunk *trunk = new Trunk(prev, prev_str);

    printTree(root->left, trunk, true);

    if (!prev)
        trunk->str = "---";
    else if (isLeft) {
        trunk->str = ".---";
        prev_str = "   |";
    } else {
        trunk->str = "`---";
        prev->str = prev_str;
    }

    showTrunks(trunk);
    std::cout << root->data << std::endl;

    if (prev)
        prev->str = prev_str;
    trunk->str = "   |";

    printTree(root->right, trunk, false);
}

Vertex *IDSP(const int array[], int l, int r) {
    if (l > r) {
        return nullptr;
    } else {
        int m = (int) ceil((double) (l + r) / 2);
        auto p = addVertex(array[m], IDSP(array, l, m - 1), IDSP(array, m + 1, r));
        return p;
    }
}


int main() {
    int n = 100;
    int arr[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = i;
    }
    Vertex *tree = IDSP(arr, 0, n - 1);
    showInfo(tree);
    std::cout << "TreeLenSum = " << treeLenSum(tree, 1) << '\n';
    printTree(tree);
    delTree(tree);
}

#include <iostream>

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
    static int sz = 0;
    if (root) {
        sz++;
        treeSize(root->left);
        treeSize(root->right);
    }
    return sz;
}

int treeHeight(Vertex *root, int hi = 0) {
    if (root) {
        hi++;
        hi = std::max(treeHeight(root->left, hi), treeHeight(root->right, hi));
    }
    return hi;
}

int treeLenSum(Vertex *root, int hi = 0) {
    static int med = 0;
    if (root) {
        hi++;
        treeLenSum(root->left, hi);
        treeLenSum(root->right, hi);
    } else {
        med += hi;
    }
    return med;
}

double treeMediumHeight(Vertex *root) {
    return (double) treeLenSum(root) / treeSize(root);
}

Vertex *addVertex(int data = rand() % 100) {
    auto *vertex = new Vertex;
    vertex->data = data;
    vertex->left = vertex->right = nullptr;
    return vertex;
}

int checkSum(Vertex *root) {
    static int sum = 0;
    if (root) {
        sum += root->data;
        checkSum(root->left);
        checkSum(root->right);
    }
    return sum;
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

Vertex *manualFill() {
    auto root = addVertex();
    root->left = addVertex();
    root->left->right = addVertex();
    root->left->right->left = addVertex();
    root->left->right->right = addVertex();
    root->left->right->right->left = addVertex();
    return root;
}

int main() {
    auto root = manualFill();
    showInfo(root);
    std::cout << '\n';
    printTree(root);
    delTree(root);
}

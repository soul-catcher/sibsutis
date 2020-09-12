#include <iomanip>
#include <random>

#include "tree.h"
#include "GTree.h"

void printTable(const Tree &a, const Tree &b) {
    using std::setw;
    std::cout << "n = 100  Size  Check Sum  Height  Medium height\n"
              << "AVL    " << setw(6) << a.size() << setw(11) << a.checkSum() << setw(8) << a.height()
              << setw(15) << a.mediumHeight() << '\n'
              << "DBD    " << setw(6) << b.size() << setw(11) << b.checkSum() << setw(8) << b.height()
              << setw(15) << b.mediumHeight() << '\n';
}

int main() {
    int n = 100;
    AvlTree avlTree;
    DbdTree dbdTree;
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    auto rng = std::default_random_engine{};
    std::shuffle(&arr[0], &arr[n], rng);
    for (auto i : arr) {
        avlTree.add(i);
        dbdTree.add(i);
    }

    std::cout << "In order traversal:\n";
    dbdTree.printLNR();
    std::cout << '\n';
    printTable(avlTree, dbdTree);
    std::cout << "Number of levels: " << dbdTree.levels() << std::endl;
    GBinaryTree gtree(dbdTree);
    gtree.start();
}

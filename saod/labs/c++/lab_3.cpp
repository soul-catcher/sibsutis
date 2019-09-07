#include "tree.h"

#include <algorithm>
#include <random>
#include <iomanip>

void showInfo(const Tree &tree) {
    std::cout << "Size: " << tree.size()
              << "\nHeight: " << tree.height()
              << "\nMedium height: " << tree.mediumHeight()
              << "\nCheck sum: " << tree.checkSum()
              << "\nIn-order traversal (LNR): ";
    tree.printLNR();
}

void printTable(const Tree &a, const Tree &b, const Tree &c) {
    using std::setw;
    std::cout << "n = 100  Size  Check Sum  Height  Medium height\n"
              << "ISDP   " << setw(6) << a.size() << setw(11) << a.checkSum() << setw(8) << a.height()
              << setw(15) << a.mediumHeight() << '\n'
              << "SDP rec" << setw(6) << b.size() << setw(11) << b.checkSum() << setw(8) << b.height()
              << setw(15) << b.mediumHeight() << '\n'
              << "SDP dbl" << setw(6) << c.size() << setw(11) << c.checkSum() << setw(8) << c.height()
              << setw(15) << c.mediumHeight() << '\n';
}

int main() {
    int n = 100;
    SdpTree recTree, doubleTree;
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    IsdpTree isdpTree(arr, n);
    auto rng = std::default_random_engine{};
    std::shuffle(&arr[0], &arr[n], rng);
    for (auto i : arr) {
        recTree.addRec(i);
        doubleTree.addDouble(i);
    }

    std::cout << "ISDP:\n";
    showInfo(isdpTree);
    std::cout << "\nRecoursive:\n";
    showInfo(recTree);
    std::cout << "\nDouble:\n";
    showInfo(doubleTree);

    std::cout << '\n';
    printTable(isdpTree, recTree, doubleTree);
}

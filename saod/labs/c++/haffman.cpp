#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <fstream>
#include <cmath>

using namespace std;

struct Node2 {
    int a;
    char c;
    Node2 *left, *right;

    Node2() {
        left = right = nullptr;
    }

    Node2(Node2 *L, Node2 *R) {
        left = L;
        right = R;
        a = L->a + R->a;
    }
};


struct MyCompare {
    bool operator()(const Node2 *l, const Node2 *r) const {
        return l->a < r->a;
    }
};


vector<bool> code;
map<char, vector<bool> > table;

void BuildTable(Node2 *root) {
    if (root->left != nullptr) {
        code.push_back(0);
        BuildTable(root->left);
    }
    if (root->right != nullptr) {
        code.push_back(1);
        BuildTable(root->right);
    }
    if (root->left == nullptr && root->right == nullptr) table[root->c] = code;
    code.pop_back();
}

void haffman() {
    int j = 0, q = 0, kol = 0;
    double ver[75], codee[75], longer = 0, H = 0;
    ifstream f("1.txt", ios::out | ios::binary);
    map<char, int> m;
    while (!f.eof()) {
        char c = (char)f.get();
        if (c != '\n' and c != '\r') {
            m[c]++;
            kol++;
        }
    }
    list<Node2 *> t;
    for (auto &itr : m) {
        Node2 *p = new Node2;
        p->c = itr.first;
        p->a = itr.second;
        ver[j] = (double) itr.second / kol;
        cout << "Probability[" << itr.first << "] = " << ver[j] << endl;
        t.push_back(p);
        j++;
    }
    j = 0;
    while (t.size() != 1) {
        t.sort(MyCompare());
        Node2 *SonL = t.front();
        t.pop_front();
        Node2 *SonR = t.front();
        t.pop_front();
        Node2 *parent = new Node2(SonL, SonR);
        t.push_back(parent);
    }
    Node2 *root = t.front();
    BuildTable(root);

    map<char, vector<bool> >::iterator it;
    vector<bool>::iterator ii;
    for (it = table.begin(); it != table.end(); it++) {
        cout << it->first << " : ";
        for (ii = table[it->first].begin(); ii != table[it->first].end(); ii++, q++) cout << (*ii);
        cout << "\n";
        codee[j] = q;
        j++;
        q = 0;
    }
    f.clear();
    f.seekg(0);
    for (j = 0; j < m.size(); j++) {
        H += ver[j] * log2l(ver[j]);
        longer += ver[j] * codee[j];
    }
    cout << "Kol-vo = " << kol << endl;
    cout << "Size = " << m.size() << endl;
    cout << "Entropy = " << H * (-1) << endl;
    cout << "Average word length = " << longer << endl;
}


int main(int argc, char *argv[]) {
    haffman();
}

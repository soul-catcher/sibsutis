#include <iostream>
#include <iomanip>
#include <fstream>  // „«ο ΰ ΅®βλ α δ ©« ¬¨
#include <vector>
#include <map>
#include <list>
#include <cmath>

using namespace std;

const int N = 4000;


struct Record {
    char fio[30];
    unsigned short int sum;
    char date[10];
    char fio_adv[22];
};

struct Node {
    Record record;
    Node *next;
};

string prompt(const string &str) {
    cout << str;
    cout << "\n> ";
    string ans;
    cin >> ans;
    return ans;
}

int strcomp(const string &str1, const string &str2, int len = -1) {
    if (len == -1) {
        len = (int)str1.length();
    }
    for (int i = 0; i < len; ++i) {
        if (str1[i] == '\0' and str2[i] == '\0') {
            return 0;
        } else if (str1[i] == ' ' and str2[i] != ' ') {
            return -1;
        } else if (str1[i] != ' ' and str2[i] == ' ') {
            return 1;
        } else if (str1[i] < str2[i]) {
            return -1;
        } else if (str1[i] > str2[i]) {
            return 1;
        }
    }
    return 0;
}

int compare(const Record &record1, const Record &record2) {
    for (int i = 0; i < 22; ++i) {
        if (record1.fio_adv[i] > record2.fio_adv[i]) {
            return -1;
        } else if (record1.fio_adv[i] < record2.fio_adv[i]) {
            return 1;
        }
    }
    if (record1.sum > record2.sum) {
        return -1;
    } else if (record1.sum < record2.sum) {
        return 1;
    }
    return 0;
}

void MakeHeap(Record *array[], int l, int r) {
    Record *x = array[l];
    int i = l;
    while (true) {
        int j = i * 2;
        if (j >= r) {
            break;
        }
        if (j < r) {
            if (compare(*array[j + 1], *array[j]) != 1) {
                ++j;
            }
        }
        if (compare(*x, *array[j]) != 1) {
            break;
        }
        array[i] = array[j];
        i = j;
    }
    array[i] = x;
}

void HeapSort(Record *array[], int n) {
    int l = (n - 1) / 2;
    while (l >= 0) {
        MakeHeap(array, l, n);
        --l;
    }
    int r = n - 1;
    while (r > 0) {
        Record *temp = array[0];
        array[0] = array[r];
        array[r] = temp;
        r--;
        MakeHeap(array, 0, r);
    }
}

void load_to_memory(Record records[]) {
    ifstream file("testBase3.dat", ios::binary);
    if (not file.is_open()) {
        cout << "Could not open file\n";
    }
    file.read((char *) records, sizeof(Record) * N);
    file.close();
}

void print_head() {
    std::cout << "Fio                            Sum    Date       Fio adv\n";
}

void print_record(Record *record) {
    std::cout << "  " << record->fio
              << "  " << setw(5) << record->sum
              << "  " << record->date
              << "  " << record->fio_adv << "\n";
}

void show_list(Record *records[], int n = N) {
    int ind = 0;
    while (true) {
        cout << "Record  ";
        print_head();
        for (int i = 0; i < 20; i++) {
            Record *record = records[ind + i];
            cout << "[" << setw(4) << ind + i + 1 << "]";
            print_record(record);

        }
        string chose = prompt("w: Next page\t"
                              "q: Last page\t"
                              "e: Skip 10 next pages\n"
                              "s: Prev page\t"
                              "a: First page\t"
                              "d: Skip 10 prev pages\n"
                              "Any key: Exit");
        switch (chose[0]) {
            case 'w':
                ind += 20;
                break;
            case 's':
                ind -= 20;
                break;
            case 'a':
                ind = 0;
                break;
            case 'q':
                ind = n - 20;
                break;
            case 'd':
                ind -= 200;
                break;
            case 'e':
                ind += 200;
                break;
            default:
                return;
        }
        if (ind < 0) {
            ind = 0;
        } else if (ind > n - 20) {
            ind = n - 20;
        }
    }
}

int quick_search(Record *arr[], const std::string &key) {
    int l = 0;
    int r = N - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (strcomp(arr[m]->fio_adv, key, 3) < 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (strcomp(arr[r]->fio_adv, key, 3) == 0) {
        return r;
    }
    return -1;
}

void make_index_array(Record arr[], Record *ind_arr[], int n = N) {
    for (int i = 0; i < n; ++i) {
        ind_arr[i] = &arr[i];
    }
}

void make_index_array(Record *arr[], Node *root, int n = N) {
    Node *p = root;
    for (int i = 0; i < n; i++) {
        arr[i] = &(p->record);
        p = p->next;
    }
}

struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
};

void SDPREC(Record *D, Vertex *&p) {   //δξαΰβλενθε β πεκσπρ
    if (!p) {
        p = new Vertex;
        p->data = D;
        p->left = nullptr;
        p->right = nullptr;
    } else if (strcomp(D->date, p->data->date) == -1) {
        SDPREC(D, p->left);
    } else if (strcomp(D->date, p->data->date) >= 0) {
        SDPREC(D, p->right);
    }
}

void A2(int L, int R, int w[], Record *V[], Vertex *&root) {
    int wes = 0, sum = 0;
    int i;
    if (L <= R) {
        for (i = L; i <= R; i++)
            wes += w[i];
        for (i = L; i <= R - 1; i++) {
            if ((sum < (wes / 2)) && ((sum + w[i]) > (wes / 2))) break;
            sum += w[i];
        }
        SDPREC(V[i - 1], root);
        A2(L, i - 1, w, V, root);
        A2(i + 1, R, w, V, root);
    }
}

void Print_tree(Vertex *p, int &i)
{
    if (p)
    {
        Print_tree(p->left, i);
        std::cout << "[" << std::setw(4) << i++ << "]";
        print_record(p->data);
        Print_tree(p->right, i);
    }
}

void search_in_tree(Vertex *root, const string &key, int &i) {
    if (root) {
        if (strcomp(key, root->data->date) == -1) {
            search_in_tree(root->left, key, i);
        } else if (strcomp(key, root->data->date) == 1) {
            search_in_tree(root->right, key, i);
        } else if (strcomp(key, root->data->date) == 0) {
            search_in_tree(root->left, key, i);
            std::cout << "[" << std::setw(4) << i++ << "]";
            print_record(root->data);
            search_in_tree(root->right, key, i);
        }
    }
}


void rmtree(Vertex *root) {
    if (root) {
        rmtree(root->right);
        rmtree(root->left);
        delete root;
    }
}

void tree(Record *arr[], int n) {
    Vertex *root = nullptr;
    int w[n + 1];
    for (int i = 0; i <= n; ++i) {
        w[i] = rand() % 100;
    }
    A2(1, n, w, arr, root);
    print_head();
    int i = 1;
    Print_tree(root, i);
    string key;
    do {
        key = prompt("Input search key, q - exit");
        print_head();
        i = 1;
        search_in_tree(root, key, i);
    } while (key[0] != 'q');
    rmtree(root);

}

void search(Record *arr[], int &ind, int &n) {
    string key;
    Node *head = nullptr, *tail = nullptr;
    do {
        key = prompt("Input search key (3 characters)");
    } while (key.length() != 3);
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        head = new Node{*arr[ind], nullptr};
        tail = head;
        int i;
        for (i = ind + 1; i < 4000 && strcomp(arr[i]->fio_adv, key, 3) == 0; ++i) {
            tail->next = new Node{*arr[i], nullptr};
            tail = tail->next;
        }
        n = i - ind;
        Record *find_arr[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
    }
}


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
    ifstream f("testBase3.dat", ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Could not open file");
    }
    Record records[N];
    f.read((char *) records, sizeof(Record) * N);
    f.close();
    map<char, int> m;
    for (auto record : records) {
        for (auto i : record.fio) {
            if (i) {
                m[i]++;
                kol++;
            }
        }
        for (auto i : record.fio_adv) {
            if (i) {
                m[i]++;
                kol++;
            }
        }
        for (auto i : record.date) {
            if (i) {
                m[i]++;
                kol++;
            }
        }
        for (auto i : to_string(record.sum)) {
            m[i]++;
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

void mainloop(Record *unsorted_ind_array[], Record *sorted_ind_array[]) {
    int search_ind, search_n = -1;
    while (true) {
        string chose = prompt("1: Show unsorted list\n"
                              "2: Show sorted list\n"
                              "3: Search\n"
                              "4: Tree\n"
                              "5: Coding\n"
                              "Any key: Exit");
        switch (chose[0]) {
            case '1':
                show_list(unsorted_ind_array);
                break;
            case '2':
                show_list(sorted_ind_array);
                break;
            case '3':
                search(sorted_ind_array, search_ind, search_n);
                break;
            case '4':
                if (search_n == -1) {
                    cout << "Please search first\n";
                } else {
                    tree(&sorted_ind_array[search_ind], search_n);
                }
                break;
            case '5':
                haffman();
                break;
            default:
                return;
        }
    }
}


int main() {
    Record records[N];
    load_to_memory(records);
    Record *unsorted_ind_arr[N];
    make_index_array(records, unsorted_ind_arr);
    Record *sorted_ind_arr[N];
    make_index_array(records, sorted_ind_arr);
    HeapSort(sorted_ind_arr, N);
    mainloop(unsorted_ind_arr, sorted_ind_arr);
}

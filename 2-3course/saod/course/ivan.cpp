#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

const int N = 4000;


struct Record {
    char fio[30];
    short int department;
    char post[22];
    char birth_date[10];

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
        len = (int) str1.length();
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

    if (record1.department > record2.department) {
        return -1;
    } else if (record1.department < record2.department) {
        return 1;
    }
    return strcomp(record1.fio, record2.fio) * -1;
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
    ifstream file("testBase2.dat", ios::binary);
    if (not file.is_open()) {
        cout << "Could not open file\n";
    }
    file.read((char *) records, sizeof(Record) * N);
    file.close();
}

void print_head() {
    std::cout << "Record  Fio                     Department  Post                   Birth date\n";
}

void print_record(Record *record) {
    std::cout << "  " << record->fio
              << "  " << setw(3) << record->department
              << "  " << record->post
              << "  " << record->birth_date << "\n";
}

void show_list(Record *records[], int n = N) {
    int ind = 0;
    while (true) {
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

int quick_search(Record *arr[], int key) {
    int l = 0;
    int r = N - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m]->department < key) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (arr[r]->department == key) {
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
    int balance;

    Vertex(Record *data, Vertex *left = nullptr, Vertex *right = nullptr, int balance = 0) :
            data(data), left(left), right(right), balance(balance) {}
};

bool grow;

void ll(Vertex *&p) {
    Vertex *q = p->left;
    p->balance = q->balance = 0;
    p->left = q->right;
    q->right = p;
    p = q;
}

void rr(Vertex *&p) {
    Vertex *q = p->right;
    p->balance = q->balance = 0;
    p->right = q->left;
    q->left = p;
    p = q;
}

void lr(Vertex *&p) {
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

void rl(Vertex *&p) {
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

void add_to_avl(Vertex *&p, Record *data) {
    if (!p) {
        p = new Vertex(data);
        grow = true;
    } else if (strcomp(p->data->birth_date, data->birth_date) == 1) {
        add_to_avl(p->left, data);
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
    } else if (strcomp(p->data->birth_date, data->birth_date) <= 0) {
        add_to_avl(p->right, data);
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
}

void Print_tree(Vertex *p, int &i) {
    if (p) {
        Print_tree(p->left, i);
        std::cout << "[" << std::setw(4) << i++ << "]";
        print_record(p->data);
        Print_tree(p->right, i);
    }
}

void search_in_tree(Vertex *root, const string &key, int &i) {
    if (root) {
        if (strcomp(key, root->data->birth_date) == -1) {
            search_in_tree(root->left, key, i);
        } else if (strcomp(key, root->data->birth_date) == 1) {
            search_in_tree(root->right, key, i);
        } else if (strcomp(key, root->data->birth_date) == 0) {
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

    for (int i = 0; i < n; ++i) {
        add_to_avl(root, arr[i]);
    }
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
    Node *head = nullptr, *tail = nullptr;
    int key;
    while (true) {
        try {
            key = stoi(prompt("Input search key depatrment"));
            break;
        } catch (invalid_argument &exc) {
            cout << "Please input a number\n";
            continue;
        }
    }
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        head = new Node{*arr[ind], nullptr};
        tail = head;
        int i;
        for (i = ind + 1; i < 4000 && arr[i]->department == key; ++i) {
            tail->next = new Node{*arr[i], nullptr};
            tail = tail->next;
        }
        n = i - ind;
        Record *find_arr[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
    }
}


void shannon(const int n, double p[], int Length[], char c[][20]) {
    double q[n];
    q[0] = 0;
    Length[0] = -floor(log2(p[0]));
    for (int i = 1; i < n; ++i) {
        q[i] = q[i - 1] + p[i - 1];
        Length[i] = -floor(log2(p[i]));
    }
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < Length[i - 1]; ++j) {
            q[i - 1] *= 2;
            c[i - 1][j] = floor(q[i - 1]) + '0';
            if (q[i - 1] >= 1) {
                q[i - 1] -= 1;
            }
        }

    }
}

unordered_map<char, int> get_char_counts_from_file(const string &file_name, int &file_size, int n = N) {
    ifstream file(file_name, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }
    char ch_arr[sizeof(Record) * n];
    file.read((char *) ch_arr, sizeof(Record) * n);
    file.close();

    unordered_map<char, int> counter_map;
    file_size = 0;
    for (auto ch : ch_arr) {
        counter_map[ch]++;
        file_size++;
    }
    return counter_map;
}

vector<pair<double, char>> calc_probabilities(const unordered_map<char, int> &counter_map, int count) {
    vector<pair<double, char>> probabilities;
    probabilities.reserve(counter_map.size());
    for (auto i : counter_map) {
        probabilities.emplace_back(make_pair((double) i.second / count, i.first));
    }
    return probabilities;
}

void coding() {
    int file_size;
    unordered_map<char, int> counter_map;
    counter_map = get_char_counts_from_file("testBase2.dat", file_size);

    auto probabilities = calc_probabilities(counter_map, file_size);
    counter_map.clear();

    sort(probabilities.begin(), probabilities.end(), greater<pair<double, char>>());
    cout << "Probabil.  char\n";
    for (auto i : probabilities) {
        cout << fixed << i.first << " | " << i.second << '\n';
    }

    const int n = (int) probabilities.size();

    auto c = new char[n][20];
    auto Length = new int[n];
    auto p = new double[n];
    for (int i = 0; i < n; ++i) {
        p[i] = probabilities[i].first;
    }

    double shen = 0;
    shannon(n, p, Length, c);
    cout << "\nShannon Code:\n";
    cout << "\nCh  Prob      Code\n";
    double avg_len = 0;
    double entropy = 0;
    for (int i = 0; i < n; i++) {
        avg_len += Length[i] * p[i];
        entropy -= p[i] * log2(p[i]);
        printf("%c | %.5lf | ", probabilities[i].second, p[i]);
        for (int j = 0; j < Length[i]; ++j) {
            printf("%c", c[i][j]);
        }
        cout << '\n';
        shen += p[i];
    }
    cout << "Average length = " << avg_len << '\n'
         << "Entropy = " << entropy << '\n'
         << "Average length < entropy + 1\n"
         << "n = " << n << endl;

}

void mainloop(Record *unsorted_ind_array[], Record *sorted_ind_array[]) {
    int search_ind, search_n = -1;
    while (true) {
        std::string chose = prompt("1: Show unsorted list\n"
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
                    std::cout << "Please search first\n";
                } else {
                    tree(&sorted_ind_array[search_ind], search_n);
                }
                break;
            case '5':
                coding();
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

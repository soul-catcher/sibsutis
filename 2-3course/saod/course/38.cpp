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


int diff(const Record &a, const Record &b) {
    int diff = 0;
    diff = a.department - b.department;
    if (diff == 0) {
        diff = strcomp(a.fio, b.fio);
    }
    return diff;
}

void qSort(Record *array[], int L, int R) {
    while (L < R) {
        Record *x = array[(L + R) / 2];
        int i = L, j = R;
        while (i < j) {
            while (diff(*array[i], *x) < 0) {
                ++i;
            }
            while (diff(*array[j], *x) > 0) {
                --j;
            }
            if (i <= j) {
                Record *temp = array[i];
                array[i] = array[j];
                array[j] = temp;
                ++i;
                --j;
            }
        }
        if (j - L < R - i) {
            qSort(array, L, j);
            L = i;
        } else {
            qSort(array, i, R);
            R = j;
        }
    }
}

void quickSort(Record *array[], const int N) {
    qSort(array, 0, N - 1);
}

void load_to_memory(Record records[]) {
    ifstream file("testBase2.dat", ios::binary);
    if (not file.is_open()) {
        cout << "Could not open file\n";
        throw runtime_error("Could not open file");
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
        for (int i = ind; i < ind + 20 && i < n; i++) {
            Record *record = records[i];
            cout << "[" << setw(4) << i + 1 << "]";
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
                ind = n / 20 * 20;
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
        }
        while (ind >= n) {
            ind -= 20;
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
};

void dbd_add(Record *data, Vertex *&p) {
    static int vr = 1;
    static int hr = 1;
    if (!p) {
        p = new Vertex{data, nullptr, nullptr, 0};
        vr = 1;
    } else if (strcomp(data->post, p->data->post) < 0) {
        dbd_add(data, p->left);
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
    } else if (strcomp(data->post, p->data->post) >= 0) {
        dbd_add(data, p->right);
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
        if (strcomp(key, root->data->post) == -1) {
            search_in_tree(root->left, key, i);
        } else if (strcomp(key, root->data->post) == 1) {
            search_in_tree(root->right, key, i);
        } else if (strcomp(key, root->data->post) == 0) {
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
        dbd_add(arr[i], root);
    }
    print_head();
    int i = 1;
    Print_tree(root, i);
    string key;
    getline(cin, key);
    do {
        cout << "Input search key (post), q - exit\n> ";
        getline(cin, key);
        if (!key.empty() && key != "q") {
            print_head();
            i = 1;
            search_in_tree(root, key, i);
        }
    } while (key[0] != 'q');
    rmtree(root);
}

void search(Record *arr[], int &ind, int &n) {
    int key;

    key = stoi(prompt("Input search key (department)"));

    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        int i;
        for (i = ind + 1; i < 4000 && arr[i]->department == key; ++i) {}
        n = i - ind;
        show_list(&arr[ind], n);
    }
}

int Med(int L, int R, double *p) {
    double SL = 0;
    for (int i = L; i <= R; i++) {
        SL = SL + p[i];
    }
    double SR = p[R];
    int m = R;
    while (SL >= SR) {
        m--;
        SL = SL - p[m];
        SR = SR + p[m];
    }
    return m;
}

void Fano(int L, int R, int k, double *p, int Length[], char c[][20]) {
    int m;
    if (L < R) {
        k++;
        m = Med(L, R, p);
        for (int i = L; i <= R; i++) {
            if (i <= m) {
                c[i][k] = '0';
                Length[i] = Length[i] + 1;
            } else {
                c[i][k] = '1';
                Length[i] = Length[i] + 1;
            }
        }
        Fano(L, m, k, p, Length, c);
        Fano(m + 1, R, k, p, Length, c);
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

    char c[n][20];
    int Length[n];
    for (auto &i : Length) {
        i = 0;
    }

    auto p = new double[n];
    for (int i = 0; i < n; ++i) {
        p[i] = probabilities[i].first;
    }

    double shen = 0;
    Fano(0, n - 1, -1, p, Length, c);
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
         << "N = " << n << "\n\n";

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
    quickSort(sorted_ind_arr, N);
    mainloop(unsorted_ind_arr, sorted_ind_arr);
}

#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

const int N = 4000;

struct Record {
    char author[12];
    char title[32];
    char publisher[16];
    short int year;
    short int num_of_page;
};

struct Node {
    Record record;
    Node *next;
};

int strcomp(const string &str1, const string &str2, int len = numeric_limits<int>::max()) {
    for (int i = 0; i < len; ++i) {
        if (str1[i] == '\0' and str2[i] == '\0') {
            return 0;
        } else if (str1[i] < str2[i]) {
            return -1;
        } else if (str1[i] > str2[i]) {
            return 1;
        }
    }
    return 0;
}

void digitalSort(Node *&head) {
    int first_f = 16;
    int sec_f = 12;
    struct Queue {
        Node *tail;
        Node *head;
    } q[256];
    int L = first_f + sec_f;
    Node *p;

    for (int j = 0; j < L; j++) {
        for (auto &i : q) {
            i.tail = i.head = nullptr;
        }
        while (head) {
            int d;
            if (j < sec_f) {
                d = 255 - abs(head->record.author[L - j - 1 - first_f]);
            } else if (j < sec_f + first_f) {
                d = 255 - abs(head->record.publisher[L - j - 1]);
            } else {
                throw out_of_range("d out of range");
            }
            if (d > 255 or d < 0) {
                throw out_of_range("Out of queue range");
            }
            p = q[d].tail;
            if (q[d].head == nullptr)
                q[d].head = head;
            else
                p->next = head;

            p = q[d].tail = head;
            head = head->next;
            p->next = nullptr;
        }
        int i;
        for (i = 0; i < 256; i++) {
            if (q[i].head != nullptr)
                break;
        }
        head = q[i].head;
        p = q[i].tail;
        for (int k = i + 1; k < 256; k++) {
            if (q[k].head != nullptr) {
                p->next = q[k].head;
                p = q[k].tail;
            }
        }
    }
}

string prompt(const string &str) {
    cout << str;
    cout << "\n> ";
    string ans;
    cin >> ans;
    return ans;
}

Node *load_to_memory() {
    ifstream file("testBase1.dat", ios::binary);
    if (not file.is_open()) {
        return nullptr;
    }

    Node *root = nullptr;
    for (int i = 0; i < N; ++i) {
        Record record;
        file.read((char *) &record, sizeof(Record));
        root = new Node{record, root};
    }
    file.close();
    return root;
}

void make_index_array(Record *arr[], Node *root, int n = N) {
    Node *p = root;
    for (int i = 0; i < n; i++) {
        arr[i] = &(p->record);
        p = p->next;
    }
}

void print_head() {
    cout << "Author       Title                           Publisher        Year  Num of pages\n";
}

void print_record(Record *record) {
    cout << record->author
         << "  " << record->title
         << "  " << record->publisher
         << "  " << record->year
         << "  " << record->num_of_page << "\n";
}

struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
};

void SDPREC(Record *D, Vertex *&p) {
    if (!p) {
        p = new Vertex;
        p->data = D;
        p->left = nullptr;
        p->right = nullptr;
    } else if (D->year < p->data->year) {
        SDPREC(D, p->left);
    } else if (D->year >= p->data->year) {
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
        SDPREC(V[i], root);
        A2(L, i - 1, w, V, root);
        A2(i + 1, R, w, V, root);
    }
}

void Print_tree(Vertex *p) {
    static int i = 1;
    if (p) {
        Print_tree(p->left);
        cout << "[" << setw(4) << i++ << "]";
        print_record(p->data);
        Print_tree(p->right);
    }
}

void search_in_tree(Vertex *root, int key) {
    int i = 1;
    while (root) {
        if (key < root->data->year) {
            root = root->left;
        } else if (key > root->data->year) {
            root = root->right;
        } else if (key == root->data->year) {
            cout << "[" << setw(4) << i++ << "]";
            print_record(root->data);
            root = root->right;
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
    Print_tree(root);

    int key;
    while (true) {
        try {
            key = stoi(prompt("Input search key year"));
            break;
        } catch (invalid_argument &exc) {
            cout << "Please input a number\n";
            continue;
        }
    }


    print_head();
    search_in_tree(root, key);
    rmtree(root);
}

void show_list(Record *ind_arr[], int n = N) {
    int ind = 0;
    while (true) {
        cout << "Record  ";
        print_head();
        for (int i = 0; i < 20; i++) {
            Record *record = ind_arr[ind + i];
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

int quick_search(Record *arr[], const string &key) {
    int l = 0;
    int r = N - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (strcomp(arr[m]->publisher, key, 3) < 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (strcomp(arr[r]->publisher, key, 3) == 0) {
        return r;
    }
    return -1;
}

void search(Record *arr[], int &ind, int &n) {
    string key;
    do {
        key = prompt("Input search key (3 characters)");
    } while (key.length() != 3);
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        int i;
        for (i = ind + 1; strcomp(arr[i]->publisher, key, 3) == 0; ++i) {}
        n = i - ind;
        show_list(&arr[ind], n);
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

unordered_map<char, int> get_char_counts_from_file(const string &file_name, int &file_size, const int n = N) {
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
    counter_map = get_char_counts_from_file("testBase1.dat", file_size);

    auto probabilities = calc_probabilities(counter_map, file_size);
    counter_map.clear();

    sort(probabilities.begin(), probabilities.end(), greater<>());
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
         << "N = " << n << '\n';

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
                coding();
                break;
            default:
                return;
        }
    }
}

void free_mem(Node *&root) {
    while (root) {
        auto p = root;
        root = root->next;
        delete p;
    }
}

int main() {
    Node *root = load_to_memory();
    if (!root) {
        cout << "File not found" << endl;
        return 1;
    }
    Record *unsorted_ind_arr[N];
    Record *sorted_ind_arr[N];
    make_index_array(unsorted_ind_arr, root);
    digitalSort(root);
    make_index_array(sorted_ind_arr, root);
    mainloop(unsorted_ind_arr, sorted_ind_arr);
    free_mem(root);
}

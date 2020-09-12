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
    char fio[32];
    char street[18];
    union {
        short int home;
        char chars[2];
    };
    short int appartament;
    char date[10];
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

Node *load_to_memory() {
    ifstream file("testBase4.dat", ios::binary);
    if (not file.is_open()) {
        return NULL;
    }

    Node *root = NULL;
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

void digitalSort(Node *&head) {
    int first_f = 18;
    int sec_f = 2;
    struct Queue {
        Node *tail;
        Node *head;
    } q[256];
    int L = first_f + sec_f;
    Node *p;

    for (int j = 0; j < L; j++) {
        for (auto &i : q) {
            i.tail = i.head = NULL;
        }
        while (head) {
            int d;
            if (j < sec_f) {
                d = abs(head->record.chars[L - j - 1 - first_f]);
            } else if (j < sec_f + first_f) {
                d = 255 - abs(head->record.street[L - j - 1]);
            }
            p = q[d].tail;
            if (q[d].head == NULL)
                q[d].head = head;
            else
                p->next = head;

            p = q[d].tail = head;
            head = head->next;
            p->next = NULL;
        }
        int i;
        for (i = 0; i < 256; i++) {
            if (q[i].head != NULL)
                break;
        }
        head = q[i].head;
        p = q[i].tail;
        for (int k = i + 1; k < 256; k++) {
            if (q[k].head != NULL) {
                p->next = q[k].head;
                p = q[k].tail;
            }
        }
    }
}

void print_head() {
    cout << "Record Full Name                        Street          Home  Apt  Date\n";
}

void print_record(Record *record, int i) {
    cout << "[" << setw(4) << i << "] "
         << record->fio
         << "  " << record->street
         << "  " << record->home
         << "  " << setw(3) << record->appartament
         << "  " << record->date << "\n";
}


void show_list(Record *ind_arr[], int n = N) {
    int ind = 0;
    while (true) {
        print_head();
        for (int i = 0; i < 20; i++) {
            Record *record = ind_arr[ind + i];
            print_record(record, ind + i + 1);

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
        if (strcomp(arr[m]->street, key, 3) < 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (strcomp(arr[r]->street, key, 3) == 0) {
        return r;
    }
    return -1;
}

void search(Record *arr[], int &ind, int &n) {
    Node *head = NULL, *tail = NULL;
    string key;
    do {
        key = prompt("Input search key (3 characters - street)");
    } while (key.length() != 3);
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        head = new Node{*arr[ind], NULL};
        tail = head;
        int i;
        for (i = ind + 1; i < 4000 and strcomp(arr[i]->street, key, 3) == 0; ++i) {
            tail->next = new Node{*arr[i], NULL};
            tail = tail->next;
        }
        n = i - ind;
        Record *find_arr[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
    }
}

struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
    int balance;
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
        p = new Vertex{data, NULL, NULL, 0};
        grow = true;
    } else if (p->data->home > data->home) {
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
    } else if (p->data->home < data->home) {
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
        if (p->data->appartament > data->appartament) {
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
        } else {
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
        }
    }
}

void Print_tree(Vertex *p, int &i) {
    if (p) {
        Print_tree(p->left, i);
        print_record(p->data, i++);
        Print_tree(p->right, i);
    }
}

void search_in_tree(Vertex *root, int key, int key2, int &i) {
    if (root) {
        if (key < root->data->home) {
            search_in_tree(root->left, key, key2, i);
        } else if (key > root->data->home) {
            search_in_tree(root->right, key, key2, i);
        } else if (key == root->data->home) {
            if (key2 < root->data->appartament) {
                search_in_tree(root->left, key, key2, i);
            } else if (key2 > root->data->appartament) {
                search_in_tree(root->right, key, key2, i);
            } else if (key2 == root->data->appartament) {
                search_in_tree(root->left, key, key2, i);
                print_record(root->data, i++);
                search_in_tree(root->right, key, key2, i);
            }
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
    Vertex *root = NULL;
    for (int i = 0; i < n; ++i) {
        add_to_avl(root, arr[i]);
    }
    print_head();
    int i = 1;
    Print_tree(root, i);

    int key, key2;
    do {
        cout << "Input search key (home), 0 - exit\n";
        cin >> key;

        i = 1;

        cout << "Input search key (apt)\n";
        cin >> key2;
        print_head();
        search_in_tree(root, key, key2, i);
    } while (key != 0);

    rmtree(root);
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
    counter_map = get_char_counts_from_file("testBase4.dat", file_size);

    auto probabilities = calc_probabilities(counter_map, file_size);
    counter_map.clear();

    sort(probabilities.begin(), probabilities.end(), greater<pair<char, int>>());
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
            default:
                return;
        }
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
}

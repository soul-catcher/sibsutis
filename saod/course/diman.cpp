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
    short int home;
    short int appartament;
    char date[10];
};

struct Node {
    Record *record;
    Node *next;
};

string prompt(const string &str) {
    cout << str;
    cout << "\n> ";
    string ans;
    cin >> ans;
    return ans;
}

void load_to_memory(Record records[]) {
    ifstream file("testBase4.dat", ios::binary);
    if (not file.is_open()) {
        cout << "Could not open file\n";
        throw runtime_error("Could not open file");
    }
    file.read((char *) records, sizeof(Record) * N);
    file.close();
}

void make_index_array(Record arr[], Record *ind_arr[], int n = N) {
    for (int i = 0; i < n; ++i) {
        ind_arr[i] = &arr[i];
    }
}

void make_index_array(Record *arr[], Node *root, int n = N) {
    Node *p = root;
    for (int i = 0; i < n; i++) {
        arr[i] = p->record;
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

int diff(const Record &a, const Record &b) {
    int diff = 0;
    diff = strcomp(a.street, b.street);
    if (diff == 0) {
        diff = a.home - b.home;
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
    Node *head = nullptr, *tail = nullptr;
    string key;
    do {
        key = prompt("Input search key (3 characters of street)");
    } while (key.length() != 3);
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        head = new Node{arr[ind], nullptr};
        tail = head;
        int i;
        for (i = ind + 1; i < 4000 and strcomp(arr[i]->street, key, 3) == 0; ++i) {
            tail->next = new Node{arr[i], nullptr};
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
};

void SDPREC(Record *D, Vertex *&p) {
    if (!p) {
        p = new Vertex;
        p->data = D;
        p->left = nullptr;
        p->right = nullptr;
    } else if (D->home < p->data->home) {
        SDPREC(D, p->left);
    } else if (D->home == p->data->home) {
        if (D->appartament < p->data->appartament) {
            SDPREC(D, p->left);
        } else {
            SDPREC(D, p->right);
        }
    } else {
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

void Print_tree(Vertex *p, int &i) {
    if (p) {
        Print_tree(p->left, i);
        print_record(p->data, i++);
        Print_tree(p->right, i);
    }
}

void search_in_tree(Vertex *root, int key1, int key2) {
    int i = 1;
    while (root) {
        if (key1 < root->data->home) {
            root = root->left;
        } else if (key1 > root->data->home) {
            root = root->right;
        } else if (key1 == root->data->home) {
            if (key2 < root->data->appartament) {
                root = root->left;
            } else if (key2 > root->data->appartament) {
                root = root->right;
            } else if (key2 == root->data->appartament) {
                print_record(root->data, i++);
                root = root->right;
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
    Vertex *root = nullptr;
    int w[n + 1];
    for (int i = 0; i <= n; ++i) {
        w[i] = rand() % 100;
    }
    A2(1, n, w, arr, root);
    print_head();
    int i = 1;
    Print_tree(root, i);

    int key1, key2;
    do {
        while (true) {
            try {
                key1 = stoi(prompt("Input search key1 (apt), 0 - exit"));
                break;
            } catch (invalid_argument &exc) {
                cout << "Please input a number\n";
                continue;
            }
        }
        while (true) {
            try {
                key2 = stoi(prompt("Input search key2 (home)"));
                break;
            } catch (invalid_argument &exc) {
                cout << "Please input a number\n";
                continue;
            }
        }
        print_head();
        search_in_tree(root, key1, key2);
    } while (key1 != 0);
    rmtree(root);
}

int up(int n, double q, double *array, double chance[]) { //находит в array место, куда вставить число q и вставляет его
    int i = 0, j = 0;                 //сдвигая вниз остальные элементы
    for (i = n - 2; i >= 1; i--) {
        if (array[i - 1] < q) array[i] = array[i - 1];
        else {
            j = i;
            break;
        }
        if ((i - 1) == 0 && chance[i - 1] < q) {
            j = 0;
            break;
        }
    }
    array[j] = q;
    return j;
}

void down(int n, int j, int Length[], char c[][20]) {//формирует кодовое слово
    char pref[20];
    for (int i = 0; i < 20; i++) pref[i] = c[j][i];
    int l = Length[j];
    for (int i = j; i < n - 2; i++) {
        for (int k = 0; k < 20; k++)
            c[i][k] = c[i+1][k];
        Length[i] = Length[i+1];
    }
    for (int i = 0; i < 20; i++) {
        c[n-2][i] = pref[i];
        c[n-1][i] = pref[i];
    }
    c[n-1][l] = '1';
    c[n-2][l] = '0';
    Length[n-1] = l + 1;
    Length[n-2] = l + 1;
}

void huffman(int n, double *array, int Length[], char c[][20], double chance[]) {
    if (n == 2) {
        c[0][0] = '0';
        Length[0] = 1;
        c[1][0] = '1';
        Length[1] = 1;
    } else {
        double q = array[n - 2] + array[n - 1];
        int j = up(n, q, array, chance); //поиск и вставка суммы
        huffman(n - 1, array, Length, c, chance);
        down(n, j, Length, c); //достраиваем код
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
    counter_map = get_char_counts_from_file("testBase4.dat", file_size);

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
    double chance_l[n];
    for (int i = 0; i < n; ++i) {
        p[i] = probabilities[i].first;
        chance_l[i] = p[i];

    }

    huffman(n, chance_l, Length, c, p);
    cout << "\nHaffmanCode:\n";
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
    }
    cout << "Average length = " << avg_len << '\n'
         << "Entropy = " << entropy << '\n'
         << "Average length < entropy + 1\n"
         << "N = " << n << "\n\n";
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
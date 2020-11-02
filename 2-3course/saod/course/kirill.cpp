#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int N = 4000;


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

int strcomp(const string &str1, const string &str2, int len = 10000000) {
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
    if (record1.year > record2.year) {
        return 1;
    } else if (record1.year < record2.year) {
        return -1;
    } else {
        return strcomp(record1.author, record2.author);
    }
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
            if (compare(*array[j + 1], *array[j]) != -1) {
                ++j;
            }
        }
        if (compare(*x, *array[j]) != -1) {
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

string prompt(const string &str) {
    cout << str;
    cout << "\n> ";
    string ans;
    cin >> ans;
    return ans;
}

void load_to_memory(Record records[]) {
    ifstream file("testBase1.dat", ios::binary);
    if (not file.is_open()) {
        cout << "Could not open file\n";
        throw runtime_error("Could not open file");
    }
    file.read((char *) records, sizeof(Record) * N);
    file.close();
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
    } else if (D->num_of_page < p->data->num_of_page) {
        SDPREC(D, p->left);
    } else if (D->num_of_page >= p->data->num_of_page) {
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
        cout << V[i]->year << endl;
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
    if (root) {
        if (root->data->num_of_page > key) {
            search_in_tree(root->left, key);
        } else if (root->data->num_of_page < key) {
            search_in_tree(root->right, key);
        } else if (root->data->num_of_page == key) {
            search_in_tree(root->left, key);
            print_record(root->data);
            search_in_tree(root->right, key);
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
    do {
        while (true) {
            try {
                key = stoi(prompt("Input search key (num_of_pages), 0 - exit"));
                break;
            } catch (invalid_argument &exc) {
                cout << "Please input a number\n";
                continue;
            }
        }
        print_head();
        search_in_tree(root, key);
    } while (key != 0);
    rmtree(root);
}

void show_list(Record *ind_arr[], int n = N) {
    int ind = 0;
    while (true) {
        cout << "Record  Author       Title                           Publisher        Year  Num of pages\n";
        for (int i = 0; i < 20; i++) {
            Record *record = ind_arr[ind + i];
            cout << "[" << setw(4) << ind + i + 1 << "]"
                 << "  " << record->author
                 << "  " << record->title
                 << "  " << record->publisher
                 << "  " << record->year
                 << "  " << record->num_of_page << "\n";
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
        if (arr[m]->year < key) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (arr[r]->year == key) {
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

void search(Record *arr[], int &ind, int &n) {
    Node *head = nullptr, *tail = nullptr;
    int key;
    while (true) {
        try {
            key = stoi(prompt("Input search key (year)"));
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
        for (i = ind + 1; arr[i]->year == key; ++i) {
            tail->next = new Node{*arr[i], nullptr};
            tail = tail->next;
        }
        n = i - ind;
        Record *find_arr[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
    }
}

void mainloop(Record *unsorted_ind_array[], Record *sorted_ind_array[]) {
    int search_ind, search_n = -1;
    while (true) {
        string chose = prompt("1: Show unsorted list\n"
                              "2: Show sorted list\n"
                              "3: Search\n"
                              "4: Tree\n"
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

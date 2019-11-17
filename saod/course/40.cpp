#include <iostream>  // Ввод-вывод на консоль
#include <iomanip>  // Форматирование вывода на консоль
#include <fstream>  // Для работы с файлами
#include <string>  // Для функции stoi (string to int)
#include <vector>
#include <map>
#include <list>
#include <cmath>
#include <unordered_map>

using namespace std;

const int N = 4000;  // Размер базы данных

// структура БД
struct Record {
    char fio[30];
    union {
        short int department;
        char chars[2];
    };
    char post[22];
    char birth_date[10];
};

// Структура для того, чтобы динамически загружать связный список
struct Node {
    Record record;
    Node *next;
};

// Функция для украшения строки приглашения (не обязательна)
// Можно поменять символ приглашения с > на что-нибудь другое
string prompt(const string &str) {
    cout << str;
    cout << "\n> ";
    string ans;
    cin >> ans;
    return ans;
}

// Загрузка БД с диска в динамическую память (в связный список)
Node *load_to_memory() {
    ifstream file("testBase2.dat", ios::binary);
    if (!file.is_open()) {
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

// Создание индексного массива
void make_index_array(Record *arr[], Node *root, int n = N) {
    Node *p = root;
    for (int i = 0; i < n; i++) {
        arr[i] = &(p->record);
        p = p->next;
    }
}

// Функция для сравнения строк, возвращает -1, если str1 < str2, 1, если str1 > str2 и 0, если они равны
int strcomp(const string &str1, const string &str2, int len = -1) {
    if (len == -1) {
        len = (int) str1.length();
    }
    for (int i = 0; i < len; ++i) {
        if (str1[i] == '\0' && str2[i] == '\0') {
            return 0;
        } else if (str1[i] == ' ' && str2[i] != ' ') {
            return -1;
        } else if (str1[i] != ' ' && str2[i] == ' ') {
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
    int first_f = 2;
    int sec_f = 30;
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
                d = 255 - abs(head->record.fio[L - j - 1 - first_f]);
            } else if (j < sec_f + first_f) {
                d = (unsigned char) head->record.chars[L - j - 1];
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

// Распечатка заголовков таблицы
void print_head() {
    cout << "Record  Fio                     Department  Post                   Birth date\n";
}

// Вывод одной "записи"
void print_record(Record *record, int i) {
    cout << "[" << setw(4) << i << "]"
         << "  " << record->fio
         << "  " << setw(3) << record->department
         << "  " << record->post
         << "  " << record->birth_date << "\n";
}

// Вывод 20 "записей" и обработка ответа пользователя
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

// Двоичный поиск
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

// Реализация поиска с построением очереди
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
        auto find_arr = new Record *[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
        delete[]find_arr;
    }
}

// Вершина дерева
struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
    int balance;

    // Конструктор для структуры
    Vertex(Record *data, Vertex *left = nullptr, Vertex *right = nullptr, int balance = 0) :
            data(data), left(left), right(right), balance(balance) {}
};

// Ниже реализация АВЛ дерева
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
    } else if (strcomp(p->data->post, data->post) == 1) {
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
    } else if (strcomp(p->data->post, data->post) <= 0) {
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

// Вывод дерева с указанной вершины (слева-направо)
void Print_tree(Vertex *p, int &i) {
    if (p) {
        Print_tree(p->left, i);
        print_record(p->data, i++);
        Print_tree(p->right, i);
    }
}

// Поиск в построенном дереве
void search_in_tree(Vertex *root, const string &key, int &i) {
    if (root) {
        if (strcomp(key, root->data->post) < 0) {
            search_in_tree(root->left, key, i);
        } else if (strcomp(key, root->data->post) > 0) {
            search_in_tree(root->right, key, i);
        } else if (strcomp(key, root->data->post) == 0) {
            search_in_tree(root->left, key, i);
            print_record(root->data, i++);
            search_in_tree(root->right, key, i);
        }
    }
}

// Очичтка памяти, которая была занята деревом
void rmtree(Vertex *root) {
    if (root) {
        rmtree(root->right);
        rmtree(root->left);
        delete root;
    }
}

// Подпункт меню (дерево)
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
        getline(cin, key);
        if (!key.empty() && key != "q") {
            print_head();
            i = 1;
            search_in_tree(root, key, i);
        }
        cout << "Input search key (post), q - exit\n> ";
    } while (key[0] != 'q');

    rmtree(root);
}

// Вычисление количества вхождения каждого символа
unordered_map<char, int> get_char_counts_from_file(const string &file_name, int n = N) {
    ifstream file(file_name, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }
    auto ch_arr = new char[sizeof(Record) * n];
    file.read((char *) ch_arr, sizeof(Record) * n);
    file.close();

    unordered_map<char, int> counter_map;
//    file_size = 0;
	for (int i = 0; i < n; i++) {
		counter_map[ch_arr[i]]++;
	}
    return counter_map;
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

void BuildTable(Node2 *root, map<char, vector<bool> > &table, vector<bool> &code) {
    if (root->left != nullptr) {
        code.push_back(0);
        BuildTable(root->left, table, code);
    }
    if (root->right != nullptr) {
        code.push_back(1);
        BuildTable(root->right, table, code);
    }
    if (root->left == nullptr && root->right == nullptr) {
        table[root->c] = code;
    }
    if (!code.empty()) {
        code.pop_back();
    }
}

void haffman() {
    int j = 0, q = 0, kol = 0;
    double ver[256], codee[256], longer = 0, H = 0;
    ifstream f("testBase2.dat", ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Can't open file");
    }
    auto records = new Record[N];
    f.read((char *) records, sizeof(Record) * N);
    f.close();
    auto m = get_char_counts_from_file("testBase2.dat");
    list<Node2 *> t;
    for (auto &itr : m) {
        kol += itr.second;
    }
    for (auto &itr : m) {
        Node2 *p = new Node2;
        p->c = itr.first;
        p->a = itr.second;
        ver[j] = (double) itr.second / kol;
        cout << "Probability[" << itr.first << "] = " << ver[j] << endl;
        t.push_back(p);
        j++;
    }
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
	map<char, vector<bool> > table;
    vector<bool> code;
    BuildTable(root, table, code);
    j = 0;
    map<char, vector<bool> >::iterator it;
    vector<bool>::iterator ii;
    for (auto it : table) {
        cout << it.first << " : ";
        for (auto coded : it.second) {
            cout << coded;
        }
        cout << "\n";
        codee[j] = it.second.size();
        j++;
    }
    f.clear();
    f.seekg(0);
    for (j = 0; j < m.size(); j++) {
        H += ver[j] * log2(ver[j]);
        longer += ver[j] * codee[j];
    }
    cout << "Kol-vo = " << kol << endl;
    cout << "Size = " << m.size() << endl;
    cout << "Entropy = " << -H << endl;
    cout << "Average word length = " << longer << endl;
}

// Основное меню
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
                    cout << "Please search anything before making tree\n";
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
    Node *root = load_to_memory();  // Загрузка БД в связный список
    if (!root) {  // Если не удалось загрузить, выводим ошибку и завершаем программу
        cout << "File not found" << endl;
        return 1;
    }
    auto unsorted_ind_arr = new Record*[N];
    auto sorted_ind_arr = new Record*[N];
    make_index_array(unsorted_ind_arr, root);  // Создание индексного массива по неотсортированному списку
    digitalSort(root);  // Сортировка списка
    make_index_array(sorted_ind_arr, root);  // Создание инлексного массива по отсортированному списку
    mainloop(unsorted_ind_arr, sorted_ind_arr);  // Запуск главного меню
}

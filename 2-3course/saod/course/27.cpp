#include <iostream>  // Ввод-вывод на консоль
#include <iomanip>  // Форматирование вывода на консоль
#include <fstream>  // Для работы с файлами
#include <vector>
#include <map>
#include <list>
#include <cmath>

using namespace std;

const int N = 4000;  // Размер базы данных

// структура БД
struct Record {
    char fio[32];
    char street[18];
    short int home;
    short int appartament;
    char date[10];
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

// Функция для сравнения строк, возвращает -1, если str1 < str2, 1, если str1 > str2 и 0, если они равны
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

// Загрузка БД с диска в динамическую память (в связный список)
Node *load_to_memory() {
    Node *root = nullptr;
    ifstream file("testBase4.dat", ios::binary);
    if (not file.is_open()) {
        return nullptr;
    }

    for (int i = 0; i < N; ++i) {
        Record record;
        file.read((char *) &record, sizeof(Record));
        root = new Node{record, root};
    }
    file.close();
    return root;
}

struct body {
    Node *head;
    Node *tail;
};

// Сравнение двух "записей" (структур) для сортировки
// Возвращает -1, если record1 > record2, 1, если меньше
bool diff(Node *a, Node *b) {
    int diff = 0;
    diff = strcomp(a->record.fio, b->record.fio);
    if (diff == 0) {
        diff = strcomp(a->record.street, b->record.street);

    }
    return diff >= 0;
}

void MergeSort(Node *&S, int n) {
    int t, q, r, i, m;
    Node *a, *b, *k, *p;
    body c[2];
    a = S;
    b = S->next;
    k = a;
    p = b;
    int x = 0;
    while (p != nullptr) {
        k->next = p->next;
        k = p;
        p = p->next;
    }
    t = 1;
    Node *temp1 = a, *temp2 = b;
    while (temp1 != nullptr) {
        temp1 = temp1->next;
    }
    while (temp2 != nullptr) {
        temp2 = temp2->next;
    }

    while (t < n) {
        c[0].tail = c[0].head = nullptr;
        c[1].tail = c[1].head = nullptr;
        i = 0;
        m = n;
        while (m > 0) {
            if (m >= t) q = t;
            else q = m;
            m = m - q;
            if (m >= t) r = t;
            else r = m;
            m = m - r;
            while (q != 0 && r != 0) {
                if (!diff(a, b)) {
                    if (c[i].tail == nullptr) {
                        c[i].tail = c[i].head = a;
                    } else {
                        c[i].tail->next = a;
                        c[i].tail = a;
                    }
                    a = a->next;
                    q--;

                } else {

                    if (c[i].tail == nullptr) {
                        c[i].tail = c[i].head = b;
                    } else {
                        c[i].tail->next = b;
                        c[i].tail = b;
                    }
                    b = b->next;
                    r--;

                }
            }
            while (q > 0) {
                if (c[i].tail == nullptr) {
                    c[i].tail = c[i].head = a;
                } else {
                    c[i].tail->next = a;
                    c[i].tail = a;
                }
                a = a->next;
                q--;
            }
            while (r > 0) {
                if (c[i].tail == nullptr) {
                    c[i].tail = c[i].head = b;
                } else {
                    c[i].tail->next = b;
                    c[i].tail = b;
                }
                b = b->next;
                r--;
            }
            i = 1 - i;
            x++;
        }
        a = c[0].head;
        b = c[1].head;
        t = 2 * t;
    }
    c[0].tail->next = nullptr;
    S = c[0].head;
}

// Распечатка заголовков таблицы
void print_head() {
    cout << "Record Full Name                       Street          Home  Apt  Date\n";
}

// Вывод одной "записи"
void print_record(Record *record) {
    cout << record->fio
         << "  " << record->street
         << "  " << record->home
         << "  " << setw(3) << record->appartament
         << "  " << record->date << "\n";
}

// Вершина дерева
struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
};

// Добавление записи в дерево рекурсивным методом
void SDPREC(Record *D, Vertex *&p) {
    if (!p) {
        p = new Vertex;
        p->data = D;
        p->left = nullptr;
        p->right = nullptr;
    } else if (D->appartament < p->data->appartament) {
        SDPREC(D, p->left);
    } else if (D->appartament >= p->data->appartament) {
        SDPREC(D, p->right);
    }
}

// Функция выстраивает дерево
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

// Вывод дерева
void Print_tree(Vertex *p) {
    static int i = 1;
    if (p) {
        Print_tree(p->left);
        cout << "[" << setw(4) << i++ << "]";
        print_record(p->data);
        Print_tree(p->right);
    }
}

// Выводит найденные по ключу записи из дерева
Record *search_in_tree(Vertex *root, int key) {
    int i = 1;
    while (root) {
        if (key < root->data->appartament) {
            root = root->left;
        } else if (key > root->data->appartament) {
            root = root->right;
        } else if (key == root->data->appartament) {
            cout << "[" << setw(4) << i++ << "]";
            print_record(root->data);
            root = root->right;
        }
    }
    return nullptr;
}

// Очищает динамическую память, занятую деревом
void rmtree(Vertex *root) {
    if (root) {
        rmtree(root->right);
        rmtree(root->left);
        delete root;
    }
}

// Меню работы с деревом
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
            key = stoi(prompt("Input search key (appartaments)"));
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

// Меню для работы с индексным массивом:
// Перемещение по списку, вывод и т.д.
void show_list(Record *ind_arr[], int n = N) {
    int ind = 0;
    while (true) {
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

// Создание индексного массива из односвязного списка
void make_index_array(Record *arr[], Node *root, int n = N) {
    Node *p = root;
    for (int i = 0; i < n; i++) {
        arr[i] = &(p->record);
        p = p->next;
    }
}

// Быстрый поиск в отсортированном массиве
int quick_search(Record *arr[], const string &key) {
    int l = 0;
    int r = N - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (strcomp(arr[m]->fio, key, 3) < 0) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (strcomp(arr[r]->fio, key, 3) == 0) {
        return r;
    }
    return -1;
}

// Меню для поиска по отсортированному индексному массиву
void search(Record *arr[], int &ind, int &n) {
    string key;
    do {
        key = prompt("Input search key (first 3 letters of fio)");
    } while (key.length() != 3);
    ind = quick_search(arr, key);
    if (ind == -1) {
        cout << "Not found\n";
    } else {
        int i;
        for (i = ind + 1; strcomp(arr[i]->fio, key, 3) == 0; ++i) {}
        n = i - ind;
        show_list(&arr[ind], n);
    }
}

// Структура для кодирования
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

// Стравнение двух структур (сверху которые)
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
    ifstream f("testBase4.dat", ios::binary);
    if (!f.is_open()) {
        throw runtime_error("Could not open file");
    }
    Record records[N];
    f.read((char *) records, sizeof(Record) * N);
    f.close();
    map<char, int> m;
    for (Record record : records) {
        for (auto i : record.fio) {
            if (i) {
                m[i]++;
                kol++;
            }
        }
        for (auto i : record.street) {
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
        for (auto i : to_string(record.home)) {
            m[i]++;
            kol++;
        }
        for (auto i : to_string(record.appartament)) {
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

// Главное меню
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
    Record *unsorted_ind_arr[N];
    Record *sorted_ind_arr[N];
    make_index_array(unsorted_ind_arr, root);  // Создание индексного массива по неотсортированному списку
    MergeSort(root, N);  // Сортировка списка
    make_index_array(sorted_ind_arr, root);  // Создание инлексного массива по отсортированному списку
    mainloop(unsorted_ind_arr, sorted_ind_arr);  // Запуск главного меню
}

#include <iostream>  // Ввод-вывод на консоль
#include <iomanip>  // Форматирование вывода на консоль
#include <fstream>  // Для работы с файлами


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

// Загрузка БД с диска в динамическую память (в связный список)
Node *load_to_memory() {
    ifstream file("testBase4.dat", ios::binary);
    if (not file.is_open()) {  // Если не удалось открыть БД, возвращаем 0
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

// Сравнение двух "записей" (структур) для сортировки
// Возвращает -1, если record1 > record2, 1, если меньше
int compare(const Record &record1, const Record &record2) {
    for (int i = 0; i < 18; ++i) {  // В цикле сравниваются улицы посимвольно
        if (record1.street[i] > record2.street[i]) {
            return -1;
        } else if (record1.street[i] < record2.street[i]) {
            return 1;
        }
    }
    if (record1.home > record2.home) {  // Сравниваются номера домов
        return -1;
    } else if (record1.home < record2.home) {
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

// Распечатка заголовков таблицы
void print_head() {
    cout << "Record Full Name                        Street          Home  Apt  Date\n";
}

// Вывод одной "записи"
void print_record(Record *record, int i) {
    cout << "[" << setw(4) << i << "] "
         << record->fio
         << "  " << record->street
         << "  " << record->home
         << "  " << setw(3) << record->appartament
         << "  " << record->date << "\n";
}

// Вывод 20 "записей" и обработка ответа пользователя
void show_list(Record *ind_arr[], int n = N) {
    int ind = 0;
    while (true) {
        system("CLS");  // Очиска экрана
        print_head();
        for (int i = 0; i < 20; i++) {  // Вывод 20 записей
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

// Функция для сравнения строк, возвращает -1, если str1 < str2, 1, если str1 > str2 и 0, если они равны
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

// Двоичный поиск
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

// Реализация поиска с построением очереди
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
        head = new Node{*arr[ind], nullptr};
        tail = head;
        int i;
        for (i = ind + 1; i < 4000 and strcomp(arr[i]->street, key, 3) == 0; ++i) {
            tail->next = new Node{*arr[i], nullptr};
            tail = tail->next;
        }
        n = i - ind;
        auto find_arr = new Record *[n];
        make_index_array(find_arr, head, n);
        show_list(find_arr, n);
        delete[] find_arr;
    }
}

// Вершина дерева
struct Vertex {
    Record *data;
    Vertex *left;
    Vertex *right;
    int balance;
};

// Добавление записи в дерево
void dbd_add(Record *data, Vertex *&p) {
    static int vr = 1;
    static int hr = 1;
    if (!p) {
        p = new Vertex{data, NULL, NULL, 0};
        vr = 1;
    } else if (strcomp(data->fio, p->data->fio) < 0) {
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
    } else if (strcomp(data->fio, p->data->fio) >= 0) {
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

// Вывод дерева на экран
void Print_tree(Vertex *p, int &i) {
    if (p) {
        Print_tree(p->left, i);
        print_record(p->data, i++);
        Print_tree(p->right, i);
    }
}

// Поиск в дереве
void search_in_tree(Vertex *root, const string &key, int &i) {
    if (root) {
        if (strcomp(key, root->data->fio) < 0) {
            search_in_tree(root->left, key, i);
        } else if (strcomp(key, root->data->fio) > 0) {
            search_in_tree(root->right, key, i);
        } else if (strcomp(key, root->data->fio) == 0) {
            search_in_tree(root->left, key, i);
            print_record(root->data, i++);
            search_in_tree(root->right, key, i);
        }
    }
}

// Очистка памяти и удаление дерева
void rmtree(Vertex *root) {
    if (root) {
        rmtree(root->right);
        rmtree(root->left);
        delete root;
    }
}

// Построение дерева по ФИО
void tree(Record *arr[], int n) {
    Vertex *root = NULL;
    for (int i = 0; i < n; ++i) {
        dbd_add(arr[i], root);
    }
    print_head();
    int i = 1;
    Print_tree(root, i);
    string key;
    do {
        key = prompt("Input fio, 0 - exit");
        print_head();
        i = 1;
        search_in_tree(root, key, i);
    } while (key != "0");
    rmtree(root);
}

// Основной цикл (главное меню программы)
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
                    cout << "Please search anything before making tree\n";
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
    Node *root = load_to_memory();  // Загрузка БД в связный список
    if (!root) {  // Если не удалось загрузить, выводим ошибку и завершаем программу
        cout << "File not found" << endl;
        return 1;
    }
    Record *unsorted_ind_arr[N];
    Record *sorted_ind_arr[N];
    make_index_array(unsorted_ind_arr, root);  // Делаем 2 индексных массива (один для
    make_index_array(sorted_ind_arr, root);    // неотсортированный, другой для сортировки) для простой
                                               // навигации по массивам
    HeapSort(sorted_ind_arr, N);  // Сортировка индексного массива
    mainloop(unsorted_ind_arr, sorted_ind_arr);  // Вызов главного меню
}

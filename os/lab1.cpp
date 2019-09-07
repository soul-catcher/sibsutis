#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

struct List {
    List *next;
    char name[20];
    double average_mark;
} *head = nullptr, *tail = nullptr;

void insert(char name[], double average_mark) {
    if (head == nullptr) {
        head = new List;
        head->next = nullptr;
        strcpy(head->name, name);
        head->average_mark = average_mark;
        tail = head;
    } else {
        tail->next = new List;
        tail = tail->next;
        tail->next = nullptr;
        strcpy(tail->name, name);
        tail->average_mark = average_mark;
    }
}

void del_first() {
    List *first = head;
    head = head->next;
    delete first;
}

void print_list() {
    List *current = head;
    while (current) {
        cout << "Name: " << current->name
             << "\naverage mark:" << current->average_mark << endl;
        current = current->next;
    }
}

void serialize() {
    ofstream ofs("file.bin", ios::binary);
    List *current = head;
    while (current) {
        ofs.write(current->name, sizeof(current->name));
        ofs.write((char*)&current->average_mark, sizeof(current->average_mark));
        current = current->next;
    }
    ofs.close();
}

void deserialize() {
    ifstream ifs("file.bin", ios::binary);
    char name[20];
    double average_mark;
    while (true) {
        ifs.read(name, sizeof(name));
        ifs.read((char*)&average_mark, sizeof(average_mark));
        if (ifs.eof()) {
            break;
        }
        insert(name, average_mark);
    }
    ifs.close();
}

int main() {
//    insert("two", 34);
//    insert("gggg", 345533);
//    insert("fads", 66777);
//    insert("three", 343);
//    serialize();
    deserialize();
    print_list();
}

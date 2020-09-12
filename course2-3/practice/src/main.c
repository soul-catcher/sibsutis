#include <ncurses.h>
#include <stdlib.h>
#include <string.h>

#include "set.h"

enum Operations {
    SUBSET, UNION, INTERSECTION, COMPLEMENT, REV_COMPLIMENT, SYMM_DIFF, REV_SUBSET
};

#define N_OPERATIONS 7


bool ch_in_str(char ch, const char str[], int ind) {
    for (int i = 0; i < ind; ++i) {
        if (str[i] == ch) {
            return true;
        }
    }
    return false;
}

int any_key() {
    printw("Press any key to continue...\n");
    int ch = getch();
    return ch == 27;
}

int get_set(char set[]) {
    int ind = 0;
    while (true) {
        int character = getch();
        if ((character >= 'a' && character <= 'z')
            && !ch_in_str(character, set, ind)) {
            printw("%c ", character);
            set[ind++] = character;
        } else if (character == 27 || character == 10) {
            set[ind] = '\0';
            printw("\n");
            return character == 27;
        } else if (character == 263 && ind > 0) {
            int x, y;
            getyx(curscr, y, x);
            mvdelch(y, x - 2);
            ind--;
        }
    }
}

int cmp(const void *a, const void *b) {
    return *(char *) a - *(char *) b;
}

int get_sets(char set_A[], char set_B[]) {
    clear();
    printw("Input first set. Elements must be unique:\n"
           "<Enter> - to the next set\n"
           "<ESC> - close program\n"
           "<Backspace> - remove last element\n\n"
           "Set A:\n");
    if (get_set(set_A)) {
        return 1;
    }
    clear();
    printw("Input second set. Elements must be unique:\n"
           "<Enter> - continue\n"
           "<ESC> - close program\n"
           "<Backspace> - remove last element\n\n"
           "Set B:\n");
    if (get_set(set_B)) {
        return 1;
    }
    qsort(set_A, strlen(set_A), sizeof(char), cmp);
    qsort(set_B, strlen(set_B), sizeof(char), cmp);
    return 0;
}

void show_menu_elements(int current_choose, const char menu[][STR_SIZE], int menu_size) {
    for (int i = 0; i < menu_size; i++) {
        if (current_choose == i) {
            attron(A_REVERSE);
            printw("> %s <\n", menu[i]);
            attroff(A_REVERSE);
        } else {
            printw("  %s\n", menu[i]);
        }
    }
}

int menu(int *choose, const char prompt[], const char menu_text[][STR_SIZE], int menu_size) {
    *choose = 0;
    curs_set(0);
    while (true) {
        clear();
        printw("%s", prompt);
        show_menu_elements(*choose, menu_text, menu_size);
        int character = getch();
        switch (character) {
            case KEY_UP:
                if (*choose > 0) {
                    (*choose)--;
                }
                break;
            case KEY_DOWN:
                if (*choose < menu_size - 1) {
                    (*choose)++;
                }
                break;
            case 27:
                curs_set(1);
                return 1;
            case 10:
                curs_set(1);
                return 0;
            default:
                break;
        }
    }
}


int get_operation(enum Operations *operation, const char operations[N_OPERATIONS][STR_SIZE]) {
    char prompt[] = "Choose operation\n<Arrow keys> - navigate\n<Esc> - close program\n\n";
    return menu((int *) operation, prompt, operations, N_OPERATIONS);
}

int show_result(char set_A[], char set_B[], enum Operations oper,
                 const char operations[N_OPERATIONS][STR_SIZE]) {
    char res[53];
    switch (oper) {
        case SUBSET:
            strcpy(res, set_subset(set_A, set_B) ? "A is subset of B" : "A is NOT subset of B");
            break;
        case UNION:
            set_union(set_A, set_B, res);
            break;
        case INTERSECTION:
            set_intersection(set_A, set_B, res);
            break;
        case COMPLEMENT:
            set_complement(set_A, set_B, res);
            break;
        case REV_COMPLIMENT:
            set_complement(set_B, set_A, res);
            break;
        case SYMM_DIFF:
            set_symm_diff(set_A, set_B, res);
            break;
        case REV_SUBSET:
            strcpy(res, set_subset(set_B, set_A) ? "B is subset of A" : "B is NOT subset of A");
            break;
    }
    clear();
    printw("Set A:\n{%s}\n\nSet B:\n{%s}\n\nOperation: %s\n\nResult:\n{%s}\n\n", set_A, set_B, operations[oper], res);
    return any_key();
}

void main_menu() {
    char set_A[27];
    char set_B[27];
    if (get_sets(set_A, set_B)) {
        return;
    }
    char operations[N_OPERATIONS][STR_SIZE] = {
            "Is A subset of B",
            "Union A and B",
            "Intersection A and B",
            "Complement of B in A",
            "Complement of A in B",
            "Symmetric difference of B and A",
            "Is B subset of A"
    };
    enum Operations oper;
    if (get_operation(&oper, operations)) {
        return;
    }
    if (show_result(set_A, set_B, oper, operations)) {
        return;
    }
    char menu_text[][STR_SIZE] = {
            "Input new sets",
            "Choose operation",
            "Show result",
            "Exit"
    };
    int user_choose;
    while (true) {
        if (menu(&user_choose, "Welcome to main menu\nChoose any action\n\n", menu_text, 4)) {
            return;
        }
        switch (user_choose) {
            case 0:
                if (get_sets(set_A, set_B)) {
                    return;
                }
                break;
            case 1:
                if (get_operation(&oper, operations)) {
                    return;
                }
                break;
            case 2:
                if (show_result(set_A, set_B, oper, operations)) {
                    return;
                }
                break;
            case 3:
                return;
            default:
                break;
        }
    }
}

int main() {
    initscr();
    keypad(stdscr, true);
    noecho();
    main_menu();
    endwin();
}

#include "set.h"


bool set_subset(const char A[], const char B[]) {
    int ind_a, ind_b;
    for (ind_a = 0, ind_b = 0; A[ind_a] != '\0' && B[ind_b] != '\0';) {
        if (A[ind_a] < B[ind_b]) {
            return false;
        } else if (A[ind_a] > B[ind_b]) {
            ind_b++;
        } else {
            ind_a++;
            ind_b++;
        }
    }
    return A[ind_a] == '\0';
}

void set_union(const char A[], const char B[], char res[]) {
    int ind_a, ind_b, ind_res;
    for (ind_a = 0, ind_b = 0, ind_res = 0; A[ind_a] != '\0' && B[ind_b] != '\0';) {
        if (A[ind_a] < B[ind_b]) {
            res[ind_res++] = A[ind_a++];
        } else if (A[ind_a] == B[ind_b]) {
            res[ind_res++] = A[ind_a++];
            ind_b++;
        } else if (A[ind_a] > B[ind_b]) {
            res[ind_res++] = B[ind_b++];
        }
    }
    while (A[ind_a] != '\0') {
        res[ind_res++] = A[ind_a++];
    }
    while (B[ind_b] != '\0') {
        res[ind_res++] = B[ind_b++];
    }
    res[ind_res] = '\0';
}

void set_intersection(const char *A, const char *B, char *res) {
    int ind_a, ind_b, ind_res;
    for (ind_a = 0, ind_b = 0, ind_res = 0; A[ind_a] != '\0' && B[ind_b] != '\0';) {
        if (A[ind_a] < B[ind_b]) {
            ind_a++;
        } else if (A[ind_a] == B[ind_b]) {
            res[ind_res++] = A[ind_a++];
            ind_b++;
        } else if (A[ind_a] > B[ind_b]) {
            ind_b++;
        }
    }
    res[ind_res] = '\0';
}

void set_complement(const char *A, const char *B, char *res) {
    int ind_a, ind_b, i;
    for (ind_a = 0, ind_b = 0, i = 0; A[ind_a] != '\0' && B[ind_b] != '\0';) {
        if (A[ind_a] < B[ind_b]) {
            res[i++] = A[ind_a++];
        } else if (A[ind_a] == B[ind_b]) {
            ind_a++;
            ind_b++;
        } else if (A[ind_a] > B[ind_b]) {
            ind_b++;
        }
    }
    while (A[ind_a] != '\0') {
        res[i++] = A[ind_a++];
    }
    res[i] = '\0';
}

void set_symm_diff(const char *A, const char *B, char *res) {
    int ind_a, ind_b, ind_res;
    for (ind_a = 0, ind_b = 0, ind_res = 0; A[ind_a] != '\0' && B[ind_b] != '\0';) {
        if (A[ind_a] < B[ind_b]) {
            res[ind_res++] = A[ind_a++];
        } else if (A[ind_a] == B[ind_b]) {
            ind_a++;
            ind_b++;
        } else if (A[ind_a] > B[ind_b]) {
            res[ind_res++] = B[ind_b++];
        }
    }
    while (A[ind_a] != '\0') {
        res[ind_res++] = A[ind_a++];
    }
    while (B[ind_b] != '\0') {
        res[ind_res++] = B[ind_b++];
    }
    res[ind_res] = '\0';
}

void set_symm_diff_dummy(const char *A, const char *B, char *res) {
    char tmp1[27], tmp2[27];
    set_complement(A, B, tmp1);
    set_complement(B, A, tmp2);
    set_union(tmp1, tmp2, res);
}


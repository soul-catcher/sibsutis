#pragma once

#include <stdbool.h>

#define STR_SIZE 100

bool set_subset(const char *A, const char *B);

void set_union(const char A[], const char B[], char res[]);

void set_intersection(const char A[], const char B[], char res[]);

void set_complement(const char A[], const char B[], char res[]);

void set_symm_diff(const char A[], const char B[], char res[]);

void set_symm_diff_dummy(const char A[], const char B[], char res[]);
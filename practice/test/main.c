#include <criterion/criterion.h>

#include "../src/set.h"

Test(subset, same) {
    char A[] = "abcd";
    char B[] = "abcd";
    cr_assert(set_subset(A, B));
}

Test(subset, one_element_true) {
    char A[] = "g";
    char B[] = "g";
    cr_assert(set_subset(A, B));
}

Test(subset, wrong) {
    char A[] = "h";
    char B[] = "g";
    cr_assert(!set_subset(A, B));
}

Test(subset, subset1) {
    char A[] = "abg";
    char B[] = "abcdefgghi";
    cr_assert(set_subset(A, B));
}

Test(subset, subset2) {
    char A[] = "a";
    char B[] = "abc";
    cr_assert(set_subset(A, B));
}

Test(subset, subset3) {
    char A[] = "c";
    char B[] = "abc";
    cr_assert(set_subset(A, B));
}

Test(subset, not_subset) {
    char A[] = "a";
    char B[] = "bcde";
    cr_assert(!set_subset(A, B));
}

Test(subset, not_subset2) {
    char A[] = "f";
    char B[] = "bcde";
    cr_assert(!set_subset(A, B));
}


Test(subset, not_subset3) {
    char A[] = "abcde";
    char B[] = "bcde";
    cr_assert(!set_subset(A, B));
}


Test(subset, not_subset4) {
    char A[] = "bcdef";
    char B[] = "bcde";
    cr_assert(!set_subset(A, B));
}

Test(subset, not_subset5) {
    char A[] = "bcdef";
    char B[] = "bcefg";
    cr_assert(!set_subset(A, B));
}

Test(subset, empty1) {
    char A[] = "\0";
    char B[] = "bcefg";
    cr_assert(set_subset(A, B));
}

Test(subset, empty2) {
    char A[] = "abcd";
    char B[] = "\0";
    cr_assert(!set_subset(A, B));
}

Test(union, simple1) {
    char A[] = "abc";
    char B[] = "def";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcdef") == 0);
}

Test(union, simple2) {
    char A[] = "abc";
    char B[] = "g";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcg") == 0);
}

Test(union, simple3) {
    char A[] = "def";
    char B[] = "abc";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcdef") == 0);
}

Test(union, test1) {
    char A[] = "abdef";
    char B[] = "c";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcdef") == 0);
}

Test(union, test2) {
    char A[] = "aceg";
    char B[] = "bcdefgh";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcdefgh") == 0);
}

Test(union, test3) {
    char A[] = "a";
    char B[] = "bcde";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "abcde") == 0);
}

Test(union, same) {
    char A[] = "aceg";
    char B[] = "aceg";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "aceg") == 0);
}

Test(union, emty1) {
    char A[] = "\0";
    char B[] = "aceg";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "aceg") == 0);
}

Test(union, empty2) {
    char A[] = "aceg";
    char B[] = "\0";
    char res[STR_SIZE];
    set_union(A, B, res);
    cr_assert(strcmp(res, "aceg") == 0);
}

Test(intersection, same) {
    char A[] = "aceg";
    char B[] = "aceg";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "aceg") == 0);
}

Test(intersection, empty) {
    char A[] = "aceg";
    char B[] = "klmn";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(intersection, simple1) {
    char A[] = "ab";
    char B[] = "bc";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "b") == 0);
}

Test(intersection, simple2) {
    char A[] = "abcde";
    char B[] = "bcde";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "bcde") == 0);
}

Test(intersection, simple3) {
    char A[] = "bc";
    char B[] = "ab";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "b") == 0);
}


Test(intersection, simple4) {
    char A[] = "bcd";
    char B[] = "ab";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "b") == 0);
}


Test(intersection, simple5) {
    char A[] = "bc";
    char B[] = "abz";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "b") == 0);
}

Test(intersection, empty1) {
    char A[] = "\0";
    char B[] = "abc";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(intersection, empty2) {
    char A[] = "abc";
    char B[] = "\0";
    char res[STR_SIZE];
    set_intersection(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(complement, same) {
    char A[] = "abcd";
    char B[] = "abcd";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(complement, simple1) {
    char A[] = "ab";
    char B[] = "bc";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "a") == 0);
}


Test(complement, simpl2) {
    char A[] = "bcdef";
    char B[] = "a";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "bcdef") == 0);
}


Test(complement, simple3) {
    char A[] = "bcdef";
    char B[] = "aghij";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "bcdef") == 0);
}


Test(complement, complex) {
    char A[] = "acefgjkoz";
    char B[] = "abcfklopq";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "egjz") == 0);
}

Test(complement, empty1) {
    char A[] = "\0";
    char B[] = "abcfklopq";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(complement, empty2) {
    char A[] = "acefgjkoz";
    char B[] = "\0";
    char res[STR_SIZE];
    set_complement(A, B, res);
    cr_assert(strcmp(res, "acefgjkoz") == 0);
}

Test(sym_diff, complex1) {
    char A[] = "aceg";
    char B[] = "bcde";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abdg") == 0);
}

Test(sym_diff, same) {
    char A[] = "abcde";
    char B[] = "abcde";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "\0") == 0);
}

Test(sym_diff, simple1) {
    char A[] = "abc";
    char B[] = "def";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcdef") == 0);
}

Test(sym_diff, simple2) {
    char A[] = "def";
    char B[] = "abc";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcdef") == 0);
}

Test(sym_diff, simple3) {
    char A[] = "abc";
    char B[] = "bcdef";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "adef") == 0);
}

Test(sym_diff, simple4) {
    char A[] = "abdeghi";
    char B[] = "cz";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcdeghiz") == 0);
}

Test(sym_diff, simple5) {
    char A[] = "bcdef";
    char B[] = "adz";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcefz") == 0);
}

Test(sym_diff, empty1) {
    char A[] = "\0";
    char B[] = "abcd";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcd") == 0);
}

Test(sym_diff, empty2) {
    char A[] = "abcd";
    char B[] = "\0";
    char res[STR_SIZE];
    set_symm_diff(A, B, res);
    cr_assert(strcmp(res, "abcd") == 0);
}
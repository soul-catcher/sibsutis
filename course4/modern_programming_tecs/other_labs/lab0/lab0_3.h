#pragma once
#include <algorithm>

int max3(int a, int b, int c) {
    return std::max({a, b, c});
}

int task2(int n) {
    int res = 0;
    for (n /= 10; n; n /= 100) {
        res = res * 10 + n % 10;
    }
    return res;
}

int min_digit(int n) {
    std::string str = std::to_string(n);
    return *std::min_element(str.begin(), str.end()) - '0';
}

int task4(const std::vector<std::vector<int>> &vec) {
    int sum = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        for (std::size_t j = 0; j < std::min(i, vec[i].size()); ++j) {
            if (vec[i][j] % 2) {
                sum += vec[i][j];
            }
        }
    }
    return sum;
}


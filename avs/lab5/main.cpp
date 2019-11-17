#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <pthread.h>
#include <x86intrin.h>

using namespace std;

void rand_fill_vec(vector<int> &vec) {
    static mt19937_64 mt_rand;
    auto uid = uniform_int_distribution<int>(1, 100);
    for (auto &elem : vec) {
        elem = uid(mt_rand);
    }
}

struct args_t {
    vector<vector<int>> *matr;
    int from, to;
};

void *rand_fill_matr_threaded(void *args) {
    auto arg = (args_t *) args;
    for (int i = arg->from; i < arg->to; ++i) {
        rand_fill_vec((*arg->matr)[i]);
    }
    return nullptr;
}

void rand_fill_matr(vector<vector<int>> &matr, int n_threads) {
    vector<pthread_t> threads(n_threads);
    vector<args_t> args;
    args.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        args.push_back({&matr,
                        static_cast<int>(static_cast<double>(matr.size()) / n_threads * (i)),
                        static_cast<int>(static_cast<double>(matr.size()) / n_threads * (i + 1))});
    }
    args[n_threads - 1].to = matr.size();
    for (int i = 0; i < n_threads; ++i) {
        pthread_create(&threads[i], nullptr, rand_fill_matr_threaded, (void *) &args[i]);
    }
    for (auto &thread : threads) {
        pthread_join(thread, nullptr);
    }
}

struct args2_t {
    vector<vector<int>> *matr;
    vector<int> *vec;
    int from, to;
    vector<int> *ans;
};

void *prod_matr_on_vect_threaded(void *args) {
    auto arg = (args2_t *) args;
    int sum = 0;
    for (int i = arg->from; i < arg->to; ++i) {
        for (int elem : (*arg->matr)[i]) {
            sum += elem * (*arg->vec)[i];
        }
        (*arg->ans)[i] = sum;
    }
    return nullptr;
}

vector<int> prod_matr_on_vect(vector<vector<int>> &matr, vector<int> &vec, int n_threads) {
    vector<pthread_t> threads(n_threads);
    vector<args2_t> args;
    vector<int> ans(matr.size());
    args.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        args.push_back({&matr,
                        &vec,
                        static_cast<int>(static_cast<double>(matr.size()) / n_threads * (i)),
                        static_cast<int>(static_cast<double>(matr.size()) / n_threads * (i + 1)),
                        &ans});
    }
    args[n_threads - 1].to = matr.size();
    for (int i = 0; i < n_threads; ++i) {
        pthread_create(&threads[i], nullptr, prod_matr_on_vect_threaded, (void *) &args[i]);
    }
    for (auto &thread : threads) {
        pthread_join(thread, nullptr);
    }
    return ans;
}

void p1() {
    int n_threads, matr_size;

    cout << "Input number of threads ";
    cin >> n_threads;
    cout << "Input matrix size ";
    cin >> matr_size;

    vector<int> vec(matr_size);
    vector<vector<int>> matr(matr_size, vector<int>(matr_size));
    rand_fill_vec(vec);
    unsigned long long start = __rdtsc();
    rand_fill_matr(matr, n_threads);
    auto res = prod_matr_on_vect(matr, vec, n_threads);
    cout << static_cast<double>(__rdtsc() - start) / 2800000000.;
    cout << "vec:\n";
    for (auto i : vec) {
        cout << i << ' ';
    }
    cout << "\nmatr:\n";
    for (auto &i : matr) {
        for (auto j : i) {
            cout << j << ' ';
        }
        cout << '\n';
    }
    cout << "res:\n";
    for (auto &i : res) {
        cout << i << ' ';
    }
}

void p2() {
    ofstream f;
    f.open("res.txt");
    for (int n_threads = 1; n_threads <= 16; ++n_threads) {
        f << "=====================> threads: " << n_threads << " <================\n";
        cout << "=====================> threads: " << n_threads << " <================\n";
        for (unsigned long long i = 10000000; i <= 900000000; i += 10000000) {
            int size = sqrt(i);
            vector<int> vec(size);
            vector<vector<int>> matr(size, vector<int>(size));

            rand_fill_vec(vec);
            unsigned long long start = __rdtsc();
            rand_fill_matr(matr, n_threads);

            auto res = prod_matr_on_vect(matr, vec, n_threads);
            double t = static_cast<double>(__rdtsc() - start) / 2800000000.;
            f << t << '\n';
            cout << t << '\n';
        }
    }
    f.close();
}

int main() {
    p2();
}
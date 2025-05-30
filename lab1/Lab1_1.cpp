#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;

void multiply_matrix(int start, int end, int n, vector < vector <int> > m1, vector < vector <int> > m2) {
    vector < vector <long> > res(end - start, vector <long>(n));
    for (int i = start; i < end; i++) {
        for (int j = 0; j < n; j++) {
            res[i - start][j] = 0;
            for (int k = 0; k < n; k++) {
                res[i - start][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    // for (int i = 0; i < end-start; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%d\t", res[i][j]);
    //     }
    //     printf("\n");
    // }
}

void treads_multiply(int n, int trd) {
    vector < vector <int> > m1(n, vector <int>(n));
    vector < vector <int> > m2(n, vector <int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1[i][j] = rand() % 100;
            m2[i][j] = rand() % 100;
        }
    }
    vector <int> lines(trd);
    for (int i = 0; i < trd; i++) {
        lines[i] = n / trd;
    }
    int s = n % trd;
    for (int i = 0; i < s; i++) {
        lines[i] += 1;
    }
    vector <int> lines_new(trd + 1, 0);
    for (int i = 1; i < trd + 1; i++) {
        lines_new[i] = lines_new[i - 1] + lines[i - 1];
    }

    vector<thread> threads;
    for (int i = 0; i < trd; i++) {
        threads.emplace_back(multiply_matrix, lines_new[i], lines_new[i + 1], n, m1, m2);
    }
    for (auto& t : threads) t.join();

    // printf("matx_1\n");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%d\t", m1[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("matx_2\n");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%d\t", m2[i][j]);
    //     }
    //     printf("\n");
    // }
}

void single_tread(int n) {
    vector < vector <long> > res(n, vector <long>(n));
    vector < vector <int> > m1(n, vector <int>(n));
    vector < vector <int> > m2(n, vector <int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m1[i][j] = rand() % 100;
            m2[i][j] = rand() % 100;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i][j] = 0;
            for (int k = 0; k < n; k++) {
                res[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

int main() {
    int thread_number = 4;
    int N[3] = { 100, 500, 1000 };
    printf("name        N        time\n");
    printf("--------------------------\n");
    for (int i = 0; i < 3; i++) {
        auto start = chrono::steady_clock::now();
        treads_multiply(N[i], thread_number);
        auto stop = chrono::steady_clock::now();
        chrono::duration<double> p = stop - start;

        start = chrono::steady_clock::now();
        single_tread(N[i]);
        stop = chrono::steady_clock::now();
        chrono::duration<double> s = stop - start;
        printf("single     %d     %4.4f s\n", N[i], s.count());
        printf("parallel   %d     %4.4f s\n", N[i], p.count());
    }

    return 0;
}

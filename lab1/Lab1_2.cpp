#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
using namespace std;

mutex mtx;
const int transactions_number = 10000;
int client_number = 4;
vector <int> BANK_ACCOUNTS = { 100, 100, 100, 100 };
vector <int> bank_accounts;
vector < vector <int> > TRANSACTIONS(transactions_number, vector <int>(2));


void transaction(int trans, int param) {
    if (param == 1) {
        bank_accounts[TRANSACTIONS[trans][0]] += TRANSACTIONS[trans][1];
    }
    else {
        mtx.lock();
        bank_accounts[TRANSACTIONS[trans][0]] += TRANSACTIONS[trans][1];
        mtx.unlock();
    };
}

void do_transactions(int trd, int param) {
    vector<thread> threads;
    int counter = 0;
    while (counter < TRANSACTIONS.size() - trd + 1) {
        vector<thread> threads;
        for (int i = counter; i < counter + trd; i++) {
            threads.emplace_back(transaction, i, param);
        }
        counter += trd;
        for (auto& t : threads) t.join();
    };
}

int main() {
    srand(time(NULL));
    for (int i = 0; i < transactions_number; i++) {
        TRANSACTIONS[i][0] = rand() % client_number;
        TRANSACTIONS[i][1] = rand() % 40 - 20;
    }

    vector <int> treads_arr = { 2, 4, 8 };

    for (int tread = 0; tread < 3; tread++) {
        cout << "Number of threads: " << treads_arr[tread] << endl;
        cout << "---------------------------------------" << endl;
        cout << "protection" << "    time     " << "result" << endl;
        cout << "---------------------------------------" << endl;

        bank_accounts = BANK_ACCOUNTS;

        auto start = chrono::steady_clock::now();
        do_transactions(treads_arr[tread], 1);
        auto stop = chrono::steady_clock::now(); 
        chrono::duration<double> t_no = stop - start;
        printf("None        %3.4f s  ",t_no.count());
        for (int i = 0; i < client_number; i++) {
            cout << bank_accounts[i] << " ";
        }
        cout << endl;

        for (int i = 0; i < client_number; i++) {
            atomic <int> a = BANK_ACCOUNTS[i];
            bank_accounts[i] = a;
        }
        start = chrono::steady_clock::now();
        do_transactions(treads_arr[tread], 1);
        stop = chrono::steady_clock::now();
        chrono::duration<double> t_atomic = stop - start;
        printf("Atomic      %3.4f s  ", t_no.count());
        for (int i = 0; i < client_number; i++) {
            cout << bank_accounts[i] << " ";
        }
        cout << endl;

        bank_accounts = BANK_ACCOUNTS;
        start = chrono::steady_clock::now();
        do_transactions(treads_arr[tread], 2);
        stop = chrono::steady_clock::now();
        chrono::duration<double> t_mutex = stop - start;
        printf("Mutex       %3.4f s  ", t_no.count());
        for (int i = 0; i < client_number; i++) {
            cout << bank_accounts[i] << " ";
        }
        cout << endl;
        cout << "=======================================" << endl;
        cout << endl;
        bank_accounts.clear();
    }

    return 0;
}

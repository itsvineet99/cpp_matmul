#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
const int N = 1024;
int main()
{
vector<vector<double>> A(N, vector<double>(N));
vector<vector<double>> B(N, vector<double>(N));
vector<vector<double>> C(N, vector<double>(N));

// Initialize matrices A and B with random values
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        A[i][j] = rand() % 100;
        B[i][j] = rand() % 100;
    }
}

auto start_serial = chrono::high_resolution_clock::now();
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
auto end_serial = chrono::high_resolution_clock::now();
auto duration_serial = chrono::duration_cast<chrono::milliseconds>(end_serial - start_serial);

std::cout << "Matmul time (ms): " << duration_serial.count() << '\n';
}

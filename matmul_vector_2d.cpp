#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
const int N = 1024;

double calculate_gflops(int m,
                        int n,
                        int k,
                        chrono::duration<double, std::milli> elapsed_ms) {
    const double elapsed_seconds = elapsed_ms.count() / 1000.0;
    if (elapsed_seconds <= 0.0) {
        return 0.0;
    }

    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    return flops / elapsed_seconds / 1e9;
}

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
chrono::duration<double, std::milli> duration_serial = end_serial - start_serial;
const double gflops = calculate_gflops(N, N, N, duration_serial);

std::cout << "Matmul time (ms): " << duration_serial.count()
          << ", GigaFLOPS: " << gflops << '\n';
}

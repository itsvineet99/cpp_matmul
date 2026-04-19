#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
const int N = 1024;

float calculate_gflops(int m,
                       int n,
                       int k,
                       chrono::duration<float, std::milli> elapsed_ms) {
    const float elapsed_seconds = elapsed_ms.count() / 1000.0f;
    if (elapsed_seconds <= 0.0f) {
        return 0.0f;
    }

    const float flops = 2.0f * static_cast<float>(m) * static_cast<float>(n) *
                        static_cast<float>(k);
    return flops / elapsed_seconds / 1e9f;
}

int main()
{
vector<vector<float>> A(N, vector<float>(N));
vector<vector<float>> B(N, vector<float>(N));
vector<vector<float>> C(N, vector<float>(N));

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
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
auto end_serial = chrono::high_resolution_clock::now();
chrono::duration<float, std::milli> duration_serial = end_serial - start_serial;
const float gflops = calculate_gflops(N, N, N, duration_serial);

std::cout << "Matmul time (ms): " << duration_serial.count()
          << ", GigaFLOPS: " << gflops << '\n';
}

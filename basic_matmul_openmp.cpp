#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
const int N = 1024;
int main()
{
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N));

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
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::milliseconds>(end_serial - start_serial);

    // Perform matrix multiplication in parallel using OpenMP
    auto start_parallel = chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        auto end_parallel = chrono::high_resolution_clock::now();
        auto duration_parallel = chrono::duration_cast<chrono::milliseconds>(end_parallel - start_parallel);

        // Display the time taken for each approach
    cout << "Time taken for serial matrix multiplication: " << duration_serial.count() << " milliseconds" << endl;
    cout << "Time taken for parallel matrix multiplication: " << duration_parallel.count() << " milliseconds" << endl;
    return 0;
}

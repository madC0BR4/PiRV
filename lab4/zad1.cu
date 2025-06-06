#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

// CPU реализация
void vectorAddCPU(const float* A, const float* B, int n) {
    for (int i = 0; i < n; ++i) {
        B[i] = A[i]*2.5;
    }
}

// CUDA-ядро
__global__ void vectorAddCUDA(const float* A, const float* B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        B[i] = A[i]*2.5;
    }
}

int main() {
    std::vector<float> h_A(N), h_B_cpu(N), h_B_gpu(N);

    // Инициализация входных массивов случайными значениями
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU сложение
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A.data(), h_B_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU время: " << cpu_time.count() << " секунд\n";

    // Выделение памяти на GPU
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU сложение
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vectorAddCUDA<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU время: " << milliseconds / 1000.0f << " секунд\n";

    // Копирование результата обратно на хост
    cudaMemcpy(h_B_gpu.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Проверка правильности
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_B_cpu[i] - h_B_gpu[i]) > 1e-5f) {
            correct = false;
            std::cout << "Ошибка на позиции " << i << ": " << h_B_cpu[i] << " != " << h_B_gpu[i] << "\n";
            break;
        }
    }
    std::cout << (correct ? "Результат корректен.\n" : "Ошибка в результате!\n");

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
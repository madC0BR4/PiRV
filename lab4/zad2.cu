#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Размер изображения
const int WIDTH = 1024;
const int HEIGHT = 1024;
const unsigned char THRESHOLD = 128;

// CPU-версия
void thresholdFilterCPU(unsigned char* input, unsigned char* output, 
                       int width, int height, unsigned char threshold) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            output[idx] = (input[idx] > threshold) ? 255 : 0;
        }
    }
}

// CUDA-ядро
__global__ void thresholdFilterCUDA(unsigned char* input, unsigned char* output, 
                                   int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

// Инициализация случайными значениями
void initializeImage(unsigned char* image, int size) {
    for (int i = 0; i < size; i++) {
        image[i] = rand() % 256;
    }
}

// Проверка результатов
bool verifyResults(unsigned char* cpu, unsigned char* gpu, int size) {
    for (int i = 0; i < size; i++) {
        if (cpu[i] != gpu[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Выделение памяти на CPU
    int imageSize = WIDTH * HEIGHT;
    unsigned char* inputImage = new unsigned char[imageSize];
    unsigned char* outputCPU = new unsigned char[imageSize];
    unsigned char* outputGPU = new unsigned char[imageSize];
    
    // Инициализация входного изображения
    initializeImage(inputImage, imageSize);
    
    // Замер времени CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    thresholdFilterCPU(inputImage, outputCPU, WIDTH, HEIGHT, THRESHOLD);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuTime = endCPU - startCPU;
    
    // Выделение памяти на GPU
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_output, imageSize * sizeof(unsigned char));
    
    // Копирование данных на GPU
    cudaMemcpy(d_input, inputImage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Настройка блоков и гридов
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Замер времени GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    thresholdFilterCUDA<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT, THRESHOLD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Копирование результатов обратно
    cudaMemcpy(outputGPU, d_output, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Проверка результатов
    if (verifyResults(outputCPU, outputGPU, imageSize)) {
        std::cout << "CPU and GPU results match!\n";
    } else {
        std::cout << "CPU and GPU results differ!\n";
    }
    
    // Вывод времени выполнения
    std::cout << "CPU time: " << cpuTime.count() * 1000 << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";
    
    // Освобождение памяти
    delete[] inputImage;
    delete[] outputCPU;
    delete[] outputGPU;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
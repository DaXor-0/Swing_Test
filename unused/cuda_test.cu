#include <stdio.h>

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
        printf("%d\t", c[idx]);
    }
}

int main() {
    const int N = 256;  // Size of vectors

    // Host vectors
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    int *h_c = (int *)malloc(N * sizeof(int));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Device vectors
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 16;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Transfer results from device to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error: mismatch at index %d, expected %d, got %d\n", i, h_a[i] + h_b[i], h_c[i]);
        }
    }

    printf("Vector addition on GPU successfully completed!\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

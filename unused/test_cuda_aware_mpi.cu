#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to perform element-wise addition of arrays
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv) {
    int rank, size;
    const int N = 10;  // Size of the array

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "This program requires at least 2 MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory on host
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    int *h_c = (int *)malloc(N * sizeof(int));
    int *h_global_c = (int *)malloc(N * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = rank + i;
        h_b[i] = size - rank + i;
    }

    // Allocate memory on device
    int *d_a, *d_b, *d_c, *d_global_c;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));
    cudaMalloc((void **)&d_global_c, N * sizeof(int));

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel on device
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Transfer results from device to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform MPI Allreduce to get global sum of h_c across all processes
    MPI_Allreduce(d_c, d_global_c, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    cudaMemcpy(h_global_c, d_global_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    // // Print results
    if (rank == 0) {
        printf("Initial arrays:\n");
        printf("h_a: ");
        for (int i = 0; i < N; ++i) {
            printf("%d ", h_a[i]);
        }
        printf("\nh_b: ");
        for (int i = 0; i < N; ++i) {
            printf("%d ", h_b[i]);
        }
        printf("\n\nResult of MPI Allreduce (global sum):\n");
        printf("h_global_c: ");
        for (int i = 0; i < N; ++i) {
            printf("%d ", h_global_c[i]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_global_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_global_c);

    MPI_Finalize();

    return 0;
}

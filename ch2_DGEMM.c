#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dgemm(int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i * n + j]; 
            for (int k = 0; k < n; k++)
                cij += A[i * n + k] * B[ k * n + j];  
            C[i * n + j] = cij;  
        }
}

// textbook
// void dgemm(int n, double* A, double* B, double* C)
// {
//     for (int i = 0; i < n; ++i)
//         for (int j = 0; j < n; ++j)
//         {
//             double cij = C[i  + j* n]; 
//             for (int k = 0; k < n; k++)
//                 cij += A[i  + k* n] * B[ k + j * n];  
//             C[i  + j* n] = cij;  
//         }
// }
int main() {
    int n = 4;  // Size of the matrix (must be a multiple of 8 for AVX-256)
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    // Memory allocation error check
    if (!A || !B || !C) {
        printf("Memory allocation failed.\n");
        return 1;
    }
    // Initialize A with unique values: A[i][j] = i + j
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = i + j;  // or use any formula like i * n + j
        }
    }

    // Initialize B with unique values: B[i][j] = i * j
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B[i * n + j] = i * j;  // or something like (i + 1) * (j + 1)
        }
    }


    // Initialize matrices with fixed values for easy validation
    for (int i = 0; i < n * n; i++) {
        // A[i] = 1.0;
        // B[i] = 2.0;
        C[i] = 0.0;
    }

    printf("Starting AVX-256 matrix multiplication...\n");

    // Measure execution time
    clock_t start = clock();
    dgemm(n, A, B, C);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f seconds\n", time_spent);

    // Print result matrix C for validation
    printf("Matrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", B[i * n + j]);
        }
        printf("\n");
    }

    printf("Result Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", C[i * n + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    printf("Program completed successfully.\n");
    return 0;
}
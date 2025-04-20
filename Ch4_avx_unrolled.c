
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#define UNROLL 4
#define N 4096
void dgemm(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; i += UNROLL * 4) {
        for (int j = 0; j < n; ++j) {
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; r++)
                c[r] = _mm256_load_pd(C + (i + r * 4) + j * n);

            for (int k = 0; k < n; k++) 
            {
                __m256d aa = _mm256_broadcastsd_pd(_mm_load_sd(A + j * n + k));
                for (int r = 0; r < UNROLL; r++)
                    c[r] = _mm256_fmadd_pd(_mm256_load_pd(B + (i + r * 4) + k * n), aa, c[r]);
            }

            for (int r = 0; r < UNROLL; r++) {
                _mm256_store_pd(C + (i + r * 4) + j * n, c[r]);
            }
        }
    }
}


void initialize_matrix(double* mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = i;
    }
}

void initialize_matrix1(double* mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = i+1;
    }
}

void transpose_matrix(double* src, double* dest, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dest[j * n + i] = src[i * n + j];
        }
    }
}

void print_matrix(const double* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f ", mat[j + i * n]);
        }
        printf("\n");
    }
}

int main() {
    int n = N;   
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));  

    
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
    clock_t start = clock();
    dgemm(n, A, B, C);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f seconds\n", time_spent);
    // printf("\nResult Matrix C:\n");
    // print_matrix(C, n);
    // Print result matrix C for validation
    
    // Free with _mm_free
    // _mm_free(A);
    // _mm_free(B);
    // _mm_free(C);
    free(A);
    free(B);
    free(C);
    return 0;
}

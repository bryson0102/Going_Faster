#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#define UNROLL 4
#define BLOCKSIZE 32
#define N 4096

void do_block(int n, int si, int sj, int sk, double* A, double* B, double* C) 
{
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4) {
        for (int j = sj; j < sj + BLOCKSIZE; j++) {
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; r++) {
                c[r] = _mm256_load_pd(C + (i + r * 4) + j * n);
            }
            for (int k = sk; k < sk + BLOCKSIZE; ++k) 
            {
                __m256d aa = _mm256_broadcastsd_pd(_mm_load_sd(A + k + j * n ));
                for (int r = 0; r < UNROLL; r++) {
                    c[r] = _mm256_fmadd_pd(_mm256_load_pd(B+(i+r*4)+k*n), aa, c[r]);
                }
            }

            for (int r = 0; r < UNROLL; r++) {
                _mm256_store_pd(C + (i + r * 4) + j * n, c[r]);
            }
        }
    }
}
void dgemm(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; i += BLOCKSIZE) 
        for (int j = 0; j < n; j += BLOCKSIZE) 
            for (int k = 0; k < n; k += BLOCKSIZE) 
                do_block(n, i, j, k, A, B, C);
}
void initialize_matrix(double* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i * n + j] = i + j;  // or use any formula like i * n + j
        }
    }
}

void initialize_matrix1(double* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i * n + j] = i * j;  // or something like (i + 1) * (j + 1)
        }
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
    
    // Initialize C to zero
    // srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        // A[i] = (double)rand() / RAND_MAX;
        // B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }
    initialize_matrix(A, n);
    initialize_matrix1(B, n);

    // printf("Matrix A:\n");
    // print_matrix(A, n);
    // printf("\nMatrix B:\n");
    // print_matrix(B, n);

    clock_t start = clock();
    dgemm(n, A, B, C);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f seconds\n", time_spent);
    // printf("\nResult Matrix C:\n");
    // print_matrix(C, n);
    free(A);
    free(B);
    free(C);
    return 0;
}

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
// #include <omp.h>
#define N 4096
#define UNROLL 4
#define BLOCKSIZE 32

void do_block(int n, int si, int sj, int sk, double* A, double* B, double* C) {
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4) {
        for (int j = sj; j < sj + BLOCKSIZE; ++j) {
            __m256d c[UNROLL];

            // Load C block
            for (int r = 0; r < UNROLL; r++) {
                c[r] = _mm256_load_pd(C + (i + r * 4) + j * n);
            }

            // Compute A * B and update C
            for (int k = sk; k < sk + BLOCKSIZE; ++k) {
                __m256d bb = _mm256_broadcast_sd(B + k * n + j);  // Corrected access pattern
                for (int r = 0; r < UNROLL; r++) {
                    __m256d aa = _mm256_load_pd(A + (i + r * 4) + k * n);
                    c[r] = _mm256_fmadd_pd(aa, bb, c[r]);  // FMA: Multiply & Add
                }
            }

            // Store result back to C
            for (int r = 0; r < UNROLL; r++) {
                _mm256_store_pd(C + (i + r * 4) + j * n, c[r]);
            }
        }
    }
}

void dgemm_parallel(int n, double* A, double* B, double* C) {
    #pragma omp parallel for
    for (int i = 0; i < n; i += BLOCKSIZE) {
        for (int j = 0; j < n; j += BLOCKSIZE) {
            for (int k = 0; k < n; k += BLOCKSIZE) {
                do_block(n, i, j, k, A, B, C);
            }
        }
    }
}

void dgemm_serial(int n, double* A, double* B, double* C) {
        for (int i = 0; i < n; i += BLOCKSIZE) {
            for (int j = 0; j < n; j += BLOCKSIZE) {
                for (int k = 0; k < n; k += BLOCKSIZE) {
                    do_block(n, i, j, k, A, B, C);
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
            printf("%.2f ", mat[j * n + i]);
        }
        printf("\n");
    }
}

int main() {
    int n = N;  
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    // double *C = (double *)malloc(n * n * sizeof(double));  
    double *C1 = (double *)malloc(n * n * sizeof(double));  // For parallel
    double *C2 = (double *)malloc(n * n * sizeof(double));  // For non-parallel

    // Initialize C to zero
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C1[i] = 0.0;
        C2[i] = 0.0;
    }
    
    clock_t start_serial = clock();
    dgemm_serial(n, A, B, C2);
    clock_t end_serial = clock();
    double time_serial = (double)(end_serial - start_serial) / CLOCKS_PER_SEC;
    printf("Serial version time: %f seconds\n", time_serial);

    clock_t start_parallel = clock();
    dgemm_parallel(n, A, B, C1);
    clock_t end_parallel = clock();
    double time_parallel = (double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;
    printf("Parallel version time: %f seconds\n", time_parallel);

    free(A);
    free(B);
    free(C1);
    free(C2);
    // free(C);
    return 0;
}

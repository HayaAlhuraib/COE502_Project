/* Standard C includes */
#include <stdlib.h>
#include <pthread.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* If we are on Darwin, include the compatibility header */
#if defined(__APPLE__)
#include "common/mach_pthread_compatibility.h"
#endif

/* Include application-specific headers */
#include "include/types.h"

#define NUM_THREADS 4  // Number of threads to use

/* Structure to hold thread data */
typedef struct {
    float* A;
    float* B;
    float* R;
    size_t rows_A;
    size_t cols_A;
    size_t cols_B;
    size_t start_row;
    size_t end_row;
} thread_data_t;

/* Function for each thread to perform matrix multiplication */
void* thread_function(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    float* A = data->A;
    float* B = data->B;
    float* R = data->R;
    size_t rows_A = data->rows_A;
    size_t cols_A = data->cols_A;
    size_t cols_B = data->cols_B;

    for (size_t i = data->start_row; i < data->end_row; i++) {
        for (size_t j = 0; j < cols_B; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            R[i * cols_B + j] = sum;
        }
    }

    return NULL;
}

/* Parallel Implementation */
void* impl_parallel(void* args) {
    args_t* arguments = (args_t*)args;
    float* input = (float*)arguments->input;
    float* output = arguments->output;
    size_t rows_A = arguments->size;
    size_t cols_A = arguments->size;
    size_t cols_B = arguments->size;

    float* A = input;
    float* B = input + rows_A * cols_A;

    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    size_t rows_per_thread = rows_A / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].R = output;
        thread_data[i].rows_A = rows_A;
        thread_data[i].cols_A = cols_A;
        thread_data[i].cols_B = cols_B;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? rows_A : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, thread_function, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return NULL;
}

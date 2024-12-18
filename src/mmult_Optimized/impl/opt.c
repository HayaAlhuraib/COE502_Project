

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"


void* impl_scalar_opt(void* args) {
    /* Extract arguments */
    args_t* arguments = (args_t*)args;
    size_t size = arguments->size;

    float* A = (float*)arguments->input;                            // Matrix A
    float* B = (float*)(arguments->input + size * size * sizeof(float)); // Matrix B
    float* R = (float*)arguments->output;                           // Result matrix R

    /* Set block size (tunable parameter) */
    size_t block_size = 16; // A typical value for cache optimization (adjust if necessary)

    /* Initialize Result Matrix */
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            R[i * size + j] = 0.0f; // Initialize all elements of R to zero
        }
    }

    /* Perform Blocked Matrix Multiplication */
    for (size_t ii = 0; ii < size; ii += block_size) {
        for (size_t jj = 0; jj < size; jj += block_size) {
            for (size_t kk = 0; kk < size; kk += block_size) {
                /* Multiply blocks */
                for (size_t i = ii; i < ii + block_size && i < size; i++) {
                    for (size_t j = jj; j < jj + block_size && j < size; j++) {
                        float sum = R[i * size + j];
                        for (size_t k = kk; k < kk + block_size && k < size; k++) {
                            sum += A[i * size + k] * B[k * size + j];
                        }
                        R[i * size + j] = sum;
                    }
                }
            }
        }
    }

    return NULL;
}


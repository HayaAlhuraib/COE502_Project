/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"


void* impl_scalar_naive(void* args) {
    /* Extract arguments */
    args_t* arguments = (args_t*)args;
    size_t size = arguments->size;

    float* A = (float*)arguments->input;                            // A starts at the beginning
    float* B = (float*)(arguments->input + size * size * sizeof(float)); // B starts after A
    float* R = (float*)arguments->output;

    /* Perform matrix-matrix multiplication */
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            R[i * size + j] = 0.0f; // Initialize the result matrix
            for (size_t k = 0; k < size; k++) {
                R[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }

    return NULL;
}


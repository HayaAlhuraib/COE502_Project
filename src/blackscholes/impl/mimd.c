/* Standard C includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "include/types.h"
#include "CNDF.h"
#include "blackscholes.h"




/* Structure to hold arguments for each thread */
typedef struct {
    float* sptPrice;
    float* strike;
    float* rate;
    float* volatility;
    float* otime;
    char* otype;
    float* output;
    size_t start;
    size_t end;
} thread_args_t;

/* Thread function to compute Black-Scholes for a subset of options */
void* thread_func(void* args) {
    thread_args_t* targs = (thread_args_t*)args;
    for (size_t i = targs->start; i < targs->end; i++) {
        float result = blackScholes(targs->sptPrice[i], targs->strike[i], targs->rate[i], targs->volatility[i], targs->otime[i], targs->otype[i]);
        if (isnan(result) || isinf(result)) {
         
            targs->output[i] = 0.0f;
        } else {
            targs->output[i] = result;
        }
    }
    return NULL;
}

/* MIMD implementation function */
void* impl_mimd(void* args) {
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;
    int nthreads = arguments->nthreads;

    pthread_t threads[nthreads];
    thread_args_t targs[nthreads];

    size_t chunk_size = num_stocks / nthreads;

    for (int i = 0; i < nthreads; i++) {
        targs[i].sptPrice   = arguments->sptPrice;
        targs[i].strike     = arguments->strike;
        targs[i].rate       = arguments->rate;
        targs[i].volatility = arguments->volatility;
        targs[i].otime      = arguments->otime;
        targs[i].otype      = arguments->otype;
        targs[i].output     = arguments->output;
        targs[i].start      = i * chunk_size;
        targs[i].end        = (i == nthreads - 1) ? num_stocks : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, thread_func, &targs[i]);
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    return NULL;
}

#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h> // For tolower
#include <time.h>

/* Include implementation headers */
#include "impl/scalar.h"
#include "impl/vec.h"
#include "impl/para.h"
#include "impl/mimd.h" 
#include "impl/simd.h" 

/* Structure to hold Black-Scholes parameters */
typedef struct {
    size_t num_stocks;   // Number of stocks
    float* sptPrice;     // Array of spot prices
    float* strike;       // Array of strike prices
    float* rate;         // Array of risk-free rates
    float* volatility;   // Array of volatilities
    float* otime;        // Array of times to maturity
    int* otype;          // Array of option types (1 for put, 0 for call)
    float* output;       // Output array for option prices
} args_t;

/* Function prototypes for different implementations */
void* impl_scalar(void* args);
void* impl_vector(void* args);
void* impl_parallel(void* args);
void* impl_mimd(void* args);
void* impl_simd(void* args);

/* Helper function to free allocated memory */
void free_args(args_t* args) {
    free(args->sptPrice);
    free(args->strike);
    free(args->rate);
    free(args->volatility);
    free(args->otime);
    free(args->otype);
    free(args->output);
}

/* Function to measure execution time of an implementation */
double measure_execution_time(void* (*impl)(void*), args_t* args, int nruns) {
    clock_t start_time = clock();
    for (int i = 0; i < nruns; i++) {
        (*impl)(args);
    }
    clock_t end_time = clock();
    return (double)(end_time - start_time) / CLOCKS_PER_SEC;
}

/* Function to generate random float within a range */
float rand_float(float min, float max) {
    return min + ((float) rand() / RAND_MAX) * (max - min);
}

int main(int argc, char** argv) {
    /* Set the buffer for printf to NULL */
    setbuf(stdout, NULL);

    /* Arguments */
    int nruns = 1; // Default number of runs
    void* (*impl)(void* args) = NULL;
    const char* impl_str = NULL;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--impl") == 0) {
            assert(++i < argc);
            if (strcmp(argv[i], "naive") == 0) {
                impl = impl_scalar;
                impl_str = "scalar";
            } else if (strcmp(argv[i], "simd") == 0) {
                impl = impl_simd;
                impl_str = "simd";
            } else if (strcmp(argv[i], "mimd") == 0) {
                impl = impl_mimd;
                impl_str = "mimd";
            } else if (strcmp(argv[i], "all") == 0) {
                impl_str = "all";
            } else {
                fprintf(stderr, "Unknown implementation: %s\n", argv[i]);
                exit(1);
            }
            continue;
        }

        if (strcmp(argv[i], "--nruns") == 0) {
            assert(++i < argc);
            nruns = atoi(argv[i]);
            continue;
        }
    }

    if (impl_str == NULL) {
        fprintf(stderr, "Usage: %s -i {naive|simd|mimd|all} [--nruns nruns]\n", argv[0]);
        exit(1);
    }

    /* Number of stocks */
    size_t num_stocks;
    printf("Enter the number of stocks: ");
    if (scanf("%zu", &num_stocks) != 1) {
        fprintf(stderr, "Error reading the number of stocks.\n");
        return 1;
    }

    /* Get number of runs */
    printf("Enter the number of runs: ");
    if (scanf("%d", &nruns) != 1) {
        fprintf(stderr, "Error reading the number of runs.\n");
        return 1;
    }

    /* Seed random number generator */
    srand(time(NULL));

    /* Allocate memory for inputs */
    float* sptPrice = malloc(num_stocks * sizeof(float));
    float* strike = malloc(num_stocks * sizeof(float));
    float* rate = malloc(num_stocks * sizeof(float));
    float* volatility = malloc(num_stocks * sizeof(float));
    float* otime = malloc(num_stocks * sizeof(float));
    int* otype = malloc(num_stocks * sizeof(int));
    float* output = malloc(num_stocks * sizeof(float));

    if (!sptPrice || !strike || !rate || !volatility || !otime || !otype || !output) {
        fprintf(stderr, "Memory allocation failed.\n");
        free_args(&(args_t){.sptPrice = sptPrice, .strike = strike, .rate = rate, 
                            .volatility = volatility, .otime = otime, .otype = otype, .output = output});
        return 1;
    }

    /* Generate random inputs for each stock */
    for (size_t i = 0; i < num_stocks; i++) {
        sptPrice[i] = rand_float(50, 150);
        strike[i] = rand_float(50, 150);
        rate[i] = rand_float(0.01, 0.05);
        volatility[i] = rand_float(0.1, 0.5);
        otime[i] = rand_float(0.5, 2);
        otype[i] = rand() % 2; // 0 for call, 1 for put
    }

    /* Create args_t structure */
    args_t args = {
        .num_stocks = num_stocks,
        .sptPrice = sptPrice,
        .strike = strike,
        .rate = rate,
        .volatility = volatility,
        .otime = otime,
        .otype = otype,
        .output = output
    };

    if (strcmp(impl_str, "all") == 0) {
        /* Measure execution times for each implementation */
        double time_naive = measure_execution_time(impl_scalar, &args, nruns);
        double time_simd = measure_execution_time(impl_simd, &args, nruns);
        double time_mimd = measure_execution_time(impl_mimd, &args, nruns);

        /* Calculate speedup */
        double speedup_simd = time_naive / time_simd;
        double speedup_mimd = time_naive / time_mimd;

        /* Print execution details */
        printf("\nExecution Times:\n");
        printf("Naive (scalar) implementation: %.6f seconds\n", time_naive);
        printf("SIMD implementation: %.6f seconds\n", time_simd);
        printf("MIMD implementation: %.6f seconds\n", time_mimd);

        printf("\nSpeedup compared to naive (scalar) implementation:\n");
        printf("SIMD speedup: %.2f\n", speedup_simd);
        printf("MIMD speedup: %.2f\n", speedup_mimd);
    } else {
        /* Measure execution time */
        double elapsed_time = measure_execution_time(impl, &args, nruns);

        /* Print results */
        printf("\nOption Prices:\n");
        for (size_t i = 0; i < num_stocks; i++) {
            printf("Stock %zu: %f\n", i + 1, output[i]);
        }

        /* Print execution details */
        printf("\nSelected implementation: %s\n", impl_str);
        printf("Number of runs: %d\n", nruns);
        printf("Execution Time: %.6f seconds\n", elapsed_time);
    }

    /* Free memory */
    free_args(&args);

    return 0;
}

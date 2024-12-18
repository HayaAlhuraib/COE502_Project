/* Standard C includes */
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

int main(int argc, char** argv) {
    /* Set the buffer for printf to NULL */
    setbuf(stdout, NULL);

    /* Arguments */
    int nruns = 1;         // Default number of runs
    void* (*impl)(void* args) = NULL;
    const char* impl_str = NULL;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--impl") == 0) {
            assert(++i < argc);
            if (strcmp(argv[i], "naive") == 0) {
                impl = impl_scalar;
                impl_str = "scalar";
            } else if (strcmp(argv[i], "vec") == 0) {
                impl = impl_vector;
                impl_str = "vector";
            } else if (strcmp(argv[i], "simd") == 0) {
                impl = impl_simd;
                impl_str = "simd";
            } else if (strcmp(argv[i], "mimd") == 0) {
                impl = impl_mimd;
                impl_str = "mimd";
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

    if (impl == NULL) {
        fprintf(stderr, "Usage: %s -i {naive|vec|simd|mimd} [--nruns nruns]\n", argv[0]);
        exit(1);
    }

    /* Number of stocks */
    size_t num_stocks;
    printf("Enter the number of stocks: ");
    if (scanf("%zu", &num_stocks) != 1) {
        fprintf(stderr, "Error reading the number of stocks.\n");
        return 1;
    }

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

    /* Get inputs for each stock */
    for (size_t i = 0; i < num_stocks; i++) {
        printf("\nEnter details for stock %zu:\n", i + 1);

        printf("Spot Price: ");
        scanf("%f", &sptPrice[i]);

        printf("Strike Price: ");
        scanf("%f", &strike[i]);

        printf("Risk-Free Rate: ");
        scanf("%f", &rate[i]);

        printf("Volatility: ");
        scanf("%f", &volatility[i]);

        printf("Time-to-Maturity: ");
        scanf("%f", &otime[i]);

        printf("Option Type ('P' for put, 'C' for call): ");
        char otype_c;
        scanf(" %c", &otype_c);
        otype[i] = (tolower(otype_c) == 'p') ? 1 : 0;
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

    /* Measure execution time */
    clock_t start_time = clock();

    for (int i = 0; i < nruns; i++) {
        (*impl)(&args); // Call the selected implementation
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    /* Print results */
    printf("\nOption Prices:\n");
    for (size_t i = 0; i < num_stocks; i++) {
        printf("Stock %zu: %f\n", i + 1, output[i]);
    }

    /* Print execution details */
    printf("\nSelected implementation: %s\n", impl_str);
    printf("Number of runs: %d\n", nruns);
    printf("Execution Time: %.6f seconds\n", elapsed_time);

    /* Free memory */
    free_args(&args);

    return 0;
}
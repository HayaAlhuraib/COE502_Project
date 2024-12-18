

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "blackscholes.h"
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"

void* impl_parallel(void* args) {
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;

    float* sptPrice   = arguments->sptPrice;
    float* strike     = arguments->strike;
    float* rate       = arguments->rate;
    float* volatility = arguments->volatility;
    float* otime      = arguments->otime;
    char* otype       = arguments->otype;
    float* output     = arguments->output;

    #pragma omp parallel for
    for (size_t i = 0; i < num_stocks; i++) {
        output[i] = blackScholes(sptPrice[i], strike[i], rate[i], volatility[i], otime[i], otype[i]);
    }

    return NULL;
}
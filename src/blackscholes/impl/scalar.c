

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */

#include "include/types.h"
#include "blackscholes.h"
#include "CNDF.h"
void* impl_scalar(void* args) { 
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;
    float* output = arguments->output;
    for (size_t i = 0; i < num_stocks; i++) {
    output[i] = blackScholes( arguments->sptPrice[i], arguments->strike[i], arguments->rate[i], arguments->volatility[i], arguments->otime[i], arguments->otype[i] );

}
return NULL; 

}
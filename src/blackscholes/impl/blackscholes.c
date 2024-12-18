
/* Standard C includes */
#include <stdlib.h>
/* Include common headers */
#include "common/macros.h"
#include "common/types.h"
#include "blackscholes.h"
#include "CNDF.h"

// Core Function: Black-Scholes Equation
float blackScholes(float spotPrice, float strike, float rate, float volatility, float time, int optionType) {
    float sqrtTime = sqrtf(time);
    float logTerm = logf(spotPrice / strike);
    float d1 = (logTerm + (rate + 0.5f * volatility * volatility) * time) / (volatility * sqrtTime);
    float d2 = d1 - volatility * sqrtTime;

    float nd1 = CNDF(d1);
    float nd2 = CNDF(d2);
    float futureValue = strike * expf(-rate * time);

    if (optionType == 0) { // Call Option
        return (spotPrice * nd1) - (futureValue * nd2);
    } else { // Put Option
        return (futureValue * (1.0f - nd2)) - (spotPrice * (1.0f - nd1));
    }
}

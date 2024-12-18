/* Standard C includes */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h> // For FLT_EPSILON to handle small floating-point values

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"
#include "blackscholes.h"
#include "CNDF.h"

// Core Function: Black-Scholes Equation
float blackScholes(float spotPrice, float strike, float rate, float volatility, float time, int optionType) {
    // Input validation
    if (spotPrice <= 0 || strike <= 0) {
        fprintf(stderr, "Error: Spot price and strike price must be positive.\n");
        return NAN;
    }

    if (volatility <= 0) {
        fprintf(stderr, "Error: Volatility must be positive.\n");
        return NAN;
    }

    if (time <= 0) {
        fprintf(stderr, "Error: Time to maturity must be positive.\n");
        return NAN;
    }

    // Calculate sqrt(time), with a check to avoid sqrt(0)
    float sqrtTime = sqrtf(time);
    if (sqrtTime <= FLT_EPSILON) {
        fprintf(stderr, "Error: sqrt(time) is too small, leading to potential division by zero.\n");
        return NAN;
    }

    // Calculate log term, with a check to avoid log(0) or log of a negative number
    if (spotPrice <= 0 || strike <= 0) {
        fprintf(stderr, "Error: Spot price and strike price must be positive for log calculation.\n");
        return NAN;
    }
    float logTerm = logf(spotPrice / strike);

    // Calculate d1 and d2
    float d1 = (logTerm + (rate + 0.5f * volatility * volatility) * time) / (volatility * sqrtTime);
    float d2 = d1 - volatility * sqrtTime;

    // Calculate CNDF values
    float nd1 = CNDF(d1);
    float nd2 = CNDF(d2);
    float futureValue = strike * expf(-rate * time);

    // Calculate option price based on option type
    if (optionType == 0) { // Call Option
        return (spotPrice * nd1) - (futureValue * nd2);
    } else if (optionType == 1) { // Put Option
        return (futureValue * (1.0f - nd2)) - (spotPrice * (1.0f - nd1));
    } else {
        fprintf(stderr, "Error: Invalid option type. Use 0 for Call, 1 for Put.\n");
        return NAN;
    }
}

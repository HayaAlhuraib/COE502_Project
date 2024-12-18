#include "CNDF.h"


float CNDF(float x) {
    int sign = (x < 0) ? 1 : 0;
    x = fabsf(x);

    float exp_val = expf(-0.5f * x * x);
    float x_nprimeofx = exp_val * 0.3989422804014327;

    float k = 1.0f / (1.0f + 0.2316419f * x);
    float k_sum = k * (0.319381530f + k * (-0.356563782f + k * (1.781477937f + 
                k * (-1.821255978f + k * 1.330274429f))));

    float result = 1.0f - (x_nprimeofx * k_sum);
    return (sign) ? 1.0f - result : result;
}
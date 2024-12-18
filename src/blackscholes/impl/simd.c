#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h> // For AVX intrinsics
#include <pthread.h>
#include "args.h" // Include the header file where args_t is defined

// Constants
#define INV_SQRT_2PI 0.3989422804014327f

// Function prototypes
void exp_simd(__m256 x, __m256 *result);
void log_simd(__m256 x, __m256 *result);
void CNDF_SIMD(__m256 x, __m256 *result);

/* SIMD CNDF Function using AVX */
void CNDF_SIMD(__m256 x, __m256 *result) {
    __m256 sign_mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.0f), _CMP_LT_OS);
    x = _mm256_andnot_ps(sign_mask, x); // absolute value

    __m256 x_squared = _mm256_mul_ps(x, x);
    __m256 exp_val;
    exp_simd(_mm256_mul_ps(_mm256_set1_ps(-0.5f), x_squared), &exp_val);

    __m256 x_nprimeofx = _mm256_mul_ps(exp_val, _mm256_set1_ps(INV_SQRT_2PI));

    __m256 k = _mm256_rcp_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_set1_ps(0.2316419f), x)));
    __m256 k_sum = _mm256_mul_ps(k, _mm256_set1_ps(0.319381530f));

    k_sum = _mm256_add_ps(_mm256_mul_ps(k, k_sum), _mm256_set1_ps(-0.356563782f));
    k_sum = _mm256_add_ps(_mm256_mul_ps(k, k_sum), _mm256_set1_ps(1.781477937f));
    k_sum = _mm256_add_ps(_mm256_mul_ps(k, k_sum), _mm256_set1_ps(-1.821255978f));
    k_sum = _mm256_add_ps(_mm256_mul_ps(k, k_sum), _mm256_set1_ps(1.330274429f));

    __m256 one_minus = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(x_nprimeofx, k_sum));

    *result = _mm256_blendv_ps(one_minus, _mm256_sub_ps(_mm256_set1_ps(1.0f), one_minus), sign_mask);
}

// Helper function for exponential (vectorized approximation)
void exp_simd(__m256 x, __m256 *result) {
    float temp[8];
    _mm256_storeu_ps(temp, x);
    for (int i = 0; i < 8; i++) {
        temp[i] = expf(temp[i]);
    }
    *result = _mm256_loadu_ps(temp);
}

// Helper function for logarithm (vectorized approximation)
void log_simd(__m256 x, __m256 *result) {
    float temp[8];
    _mm256_storeu_ps(temp, x);
    for (int i = 0; i < 8; i++) {
        temp[i] = logf(temp[i]);
    }
    *result = _mm256_loadu_ps(temp);
}


/* SIMD implementation function */
void* impl_simd(void* args) {
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;

    // Process 8 stocks at a time using SIMD
    size_t i;
    for (i = 0; i + 7 < num_stocks; i += 8) {
        // Load inputs into AVX registers
        __m256 spot_price = _mm256_loadu_ps(&arguments->sptPrice[i]);
        __m256 strike = _mm256_loadu_ps(&arguments->strike[i]);
        __m256 rate = _mm256_loadu_ps(&arguments->rate[i]);
        __m256 volatility = _mm256_loadu_ps(&arguments->volatility[i]);
        __m256 otime = _mm256_loadu_ps(&arguments->otime[i]);
        __m256 otype = _mm256_loadu_ps((float*)&arguments->otype[i]);

        // Calculate d1 and d2
        __m256 sqrt_time = _mm256_sqrt_ps(otime);
        __m256 log_term;
        log_simd(_mm256_div_ps(spot_price, strike), &log_term);

        __m256 half_vol_squared = _mm256_mul_ps(volatility, volatility);
        __m256 d1 = _mm256_div_ps(
            _mm256_add_ps(
                log_term,
                _mm256_add_ps(
                    _mm256_mul_ps(rate, otime),
                    _mm256_mul_ps(half_vol_squared, otime)
                )
            ),
            _mm256_mul_ps(volatility, sqrt_time)
        );

        __m256 d2 = _mm256_sub_ps(d1, _mm256_mul_ps(volatility, sqrt_time));

        // Calculate option prices using CNDF
        __m256 call_price = _mm256_sub_ps(
            _mm256_mul_ps(spot_price, CNDF_SIMD(d1)),
            _mm256_mul_ps(strike, _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), rate), otime), CNDF_SIMD(d2))
        );

        // Store results
        _mm256_storeu_ps(&arguments->output[i], call_price);
    }

    // Handle remaining stocks
    for (; i < num_stocks; i++) {
        float d1 = (log(arguments->sptPrice[i] / arguments->strike[i]) + 
                    (arguments->rate[i] + 0.5 * arguments->volatility[i] * arguments->volatility[i]) * arguments->otime[i]) / 
                    (arguments->volatility[i] * sqrt(arguments->otime[i]));
        float d2 = d1 - arguments->volatility[i] * sqrt(arguments->otime[i]);

        float call_price = arguments->sptPrice[i] * CNDF(d1) - 
                           arguments->strike[i] * exp(-arguments->rate[i] * arguments->otime[i]) * CNDF(d2);

        arguments->output[i] = call_price;
    }

    return NULL;
}

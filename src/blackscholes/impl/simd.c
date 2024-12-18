/* Standard C includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h> // For AVX intrinsics
#include "include/types.h"
#include "CNDF.h"
#include "blackscholes.h"

#define INV_SQRT_2PI 0.3989422804014327

// Helper function for exponential (approximated scalar function applied to vectors)
void exp_simd(__m256 x, __m256 *result) {
    float temp[8];
    _mm256_storeu_ps(temp, x);
    for (int i = 0; i < 8; i++) {
        temp[i] = expf(temp[i]);
    }
    *result = _mm256_loadu_ps(temp);
}

// Helper function for logarithm (applied element-wise)
void log_simd(__m256 x, __m256 *result) {
    float temp[8];
    _mm256_storeu_ps(temp, x);
    for (int i = 0; i < 8; i++) {
        temp[i] = logf(temp[i]);
    }
    *result = _mm256_loadu_ps(temp);
}

// SIMD CNDF Function using AVX
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

/* SIMD implementation function */
void* impl_simd(void* args) {
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;

    // Perform SIMD computations for each chunk
    for (int i = 0; i < num_stocks; i += 8) {
        // Load inputs into AVX registers
        __m256 spot_price = _mm256_loadu_ps(&arguments->sptPrice[i]);
        __m256 strike = _mm256_loadu_ps(&arguments->strike[i]);
        __m256 rate = _mm256_loadu_ps(&arguments->rate[i]);
        __m256 volatility = _mm256_loadu_ps(&arguments->volatility[i]);
        __m256 time = _mm256_loadu_ps(&arguments->otime[i]);

        __m256 sqrt_time = _mm256_sqrt_ps(time);
        __m256 log_term;
        log_simd(_mm256_div_ps(spot_price, strike), &log_term);

        __m256 d1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(rate, time), _mm256_set1_ps(0.5f)), _mm256_mul_ps(volatility, volatility));
        d1 = _mm256_add_ps(d1, log_term);
        d1 = _mm256_div_ps(d1, _mm256_mul_ps(volatility, sqrt_time));

        __m256 d2 = _mm256_sub_ps(d1, _mm256_mul_ps(volatility, sqrt_time));

        __m256 nd1, nd2;
        CNDF_SIMD(d1, &nd1);
        CNDF_SIMD(d2, &nd2);

        // Convert AVX vectors to scalar arrays
        float nd1_array[8], nd2_array[8], future_value[8];
        _mm256_storeu_ps(nd1_array, nd1);
        _mm256_storeu_ps(nd2_array, nd2);

        float temp[8];
        _mm256_storeu_ps(temp, strike);
        for (int j = 0; j < 8; j++) {
            future_value[j] = temp[j] * expf(-arguments->rate[i + j] * arguments->otime[i + j]);
        }

        // Compute results based on option type
        for (int j = 0; j < 8; j++) {
            if (arguments->otype[i + j] == 0) { // Call option
                arguments->output[i + j] = (arguments->sptPrice[i + j] * nd1_array[j]) - (future_value[j] * nd2_array[j]);
            } else { // Put option
                arguments->output[i + j] = (future_value[j] * (1.0f - nd2_array[j])) - (arguments->sptPrice[i + j] * (1.0f - nd1_array[j]));
            }
        }
    }

    return NULL;
}

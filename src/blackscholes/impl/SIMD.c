/* Standard C includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "include/types.h"
#include "CNDF.h"
#include "blackscholes.h"

#include <arm_neon.h> // For SIMD on ARM (NEON)

#define INV_SQRT_2PI 0.3989422804014327


// Helper function for exponential (approximated scalar function applied to vectors)
void exp_simd(float32x4_t x, float32x4_t *result) {
    float temp[4];
    vst1q_f32(temp, x);
    for (int i = 0; i < 4; i++) {
        temp[i] = expf(temp[i]);
    }
    *result = vld1q_f32(temp);
}

// Helper function for logarithm (applied element-wise)
void log_simd(float32x4_t x, float32x4_t *result) {
    float temp[4];
    vst1q_f32(temp, x);
    for (int i = 0; i < 4; i++) {
        temp[i] = logf(temp[i]);
    }
    *result = vld1q_f32(temp);
}
// SIMD CNDF Function using NEON
void CNDF_SIMD(float32x4_t x, float32x4_t *result) {
    float32x4_t sign_mask = vcltq_f32(x, vdupq_n_f32(0.0f));
    x = vabsq_f32(x);

    float32x4_t x_squared = vmulq_f32(x, x);
    float32x4_t exp_val;
    exp_simd(vmulq_f32(vdupq_n_f32(-0.5f), x_squared), &exp_val);

    float32x4_t x_nprimeofx = vmulq_f32(exp_val, vdupq_n_f32(INV_SQRT_2PI));

    float32x4_t k = vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), vmulq_f32(vdupq_n_f32(0.2316419f), x)));
    float32x4_t k_sum = vmulq_f32(k, vdupq_n_f32(0.319381530f));

    k_sum = vmlaq_f32(k_sum, vmulq_f32(k, k), vdupq_n_f32(-0.356563782f));
    k_sum = vmlaq_f32(k_sum, vmulq_f32(k, vmulq_f32(k, k)), vdupq_n_f32(1.781477937f));
    k_sum = vmlaq_f32(k_sum, vmulq_f32(k, vmulq_f32(k, vmulq_f32(k, k))), vdupq_n_f32(-1.821255978f));
    k_sum = vmlaq_f32(k_sum, vmulq_f32(k, vmulq_f32(k, vmulq_f32(k, vmulq_f32(k, k)))), vdupq_n_f32(1.330274429f));

    float32x4_t one_minus = vsubq_f32(vdupq_n_f32(1.0f), vmulq_f32(x_nprimeofx, k_sum));

    *result = vbslq_f32(vreinterpretq_u32_f32(sign_mask), vsubq_f32(vdupq_n_f32(1.0f), one_minus), one_minus);
}


/* SIMD implementation function */
void* impl_simd(void* args) void* impl_simd(void* args)  {
    args_t* arguments = (args_t*)args;
    size_t num_stocks = arguments->num_stocks;

    // Perform SIMD computations for each chunk
    for (int i = 0; i < num_stocks; i += 4) {
        // Load inputs into NEON registers
        float32x4_t spot_price = vld1q_f32(&arguments->sptPrice[i]);
        float32x4_t strike = vld1q_f32(&arguments->strike[i]);
        float32x4_t rate = vld1q_f32(&arguments->rate[i]);
        float32x4_t volatility = vld1q_f32(&arguments->volatility[i]);
        float32x4_t time = vld1q_f32(&arguments->otime[i]);

        float32x4_t sqrt_time = vsqrtq_f32(time);
        float32x4_t log_term;
        log_simd(vdivq_f32(spot_price, strike), &log_term);

        float32x4_t d1 = vaddq_f32(vmulq_f32(rate, time), vmulq_f32(vdupq_n_f32(0.5f), vmulq_f32(volatility, volatility)));
        d1 = vaddq_f32(d1, log_term);
        d1 = vdivq_f32(d1, vmulq_f32(volatility, sqrt_time));

        float32x4_t d2 = vsubq_f32(d1, vmulq_f32(volatility, sqrt_time));

        float32x4_t nd1, nd2;
        CNDF_SIMD(d1, &nd1);
        CNDF_SIMD(d2, &nd2);

        // Convert NEON vectors to scalar arrays
        float nd1_array[4], nd2_array[4], future_value[4];
        vst1q_f32(nd1_array, nd1);
        vst1q_f32(nd2_array, nd2);

        float temp[4];
        vst1q_f32(temp, strike);
        for (int j = 0; j < 4; j++) {
            future_value[j] = temp[j] * expf(-arguments->rate[i + j] * arguments->otime[i + j]);
        }

        // Compute results based on option type
        for (int j = 0; j < 4; j++) {
            if (arguments->otype[i + j] == 0) { // Call option
                arguments->output[i + j] = (arguments->sptPrice[i + j] * nd1_array[j]) - (future_value[j] * nd2_array[j]);
            } else { // Put option
                arguments->output[i + j] = (future_value[j] * (1.0f - nd2_array[j])) - (arguments->sptPrice[i + j] * (1.0f - nd1_array[j]));
            }
        }
    }

    return NULL;
}


#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

#include <math.h> 
#include <stdio.h> 


/* Black-Scholes function prototype */
float blackScholes(float spotPrice, float strike, float rate, float volatility, float time, int optionType);

#endif // BLACKSCHOLES_H

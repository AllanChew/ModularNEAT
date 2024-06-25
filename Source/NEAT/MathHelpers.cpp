/*
 MIT License

 Copyright (c) 2024 Allan Chew

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#include "MathHelpers.h"
#include <utility>

float NEATMathHelpers::clamp(const float& val, const float& min_val, const float& max_val) {
	if (val < min_val) return min_val;
	if (val > max_val) return max_val;
	return val;
}

int NEATMathHelpers::clamp(const int& val, const int& min_val, const int& max_val) {
	if (val < min_val) return min_val;
	if (val > max_val) return max_val;
	return val;
}

int NEATMathHelpers::lerp(const int& a, const int& b, const float& alpha) {
	return a + (alpha * (b - a));
}

double NEATMathHelpers::rand_norm() {
	return ((double)rand() / RAND_MAX);
}

// between 0 and max (inclusive)
int NEATMathHelpers::rand_int(int max) {
	const int ret_val = (max + 1) * rand_norm();
	if (ret_val > max) return max;
	return ret_val;
}

int NEATMathHelpers::rand_int(int min, int max) {
	return rand_int(max - min) + min;
}

double NEATMathHelpers::randomGaussian(double stdDev) {
	double u, v, s;

	do
	{
		u = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
		v = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
		s = u * u + v * v;
	} while (s >= 1 || s == 0);

	s = sqrt(-2.0 * log(s) / s);
	return stdDev * u * s;
}
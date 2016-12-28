#include "basic/math_function.h"

namespace surfing
{
	template <typename Dtype>
	void surfing_set(const int N, const Dtype alpha, Dtype* Y)
	{
		for (int i = 0; i < N; i++)
		{
			Y[i] = alpha;
		}
	}
	template void surfing_set<float>(const int N, const float alpha, float* Y);
	template void surfing_set<double>(const int N, const double alpha, double* Y);
}
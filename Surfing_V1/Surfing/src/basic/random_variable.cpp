#include "basic/random_variable.h"
#include "basic/math_function.h"

#include <cublas_v2.h>
#include <curand.h>
#include <random>

#include "basic/common.h"

namespace surfing
{
	template<>
	void Gaussian<float>(size_t count, float* data, float mean, float stddev)
	{
		curandGenerator_t curand_;

		CURAND_CHECK(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_MT19937));
		
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_, clock()));
		CURAND_CHECK(curandGenerateNormal(curand_, data, count, mean, stddev));

		CURAND_CHECK(curandDestroyGenerator(curand_));
	}
	template<>
	void Gaussian<double>(size_t count, double *data, double mean, double stddev)
	{
		curandGenerator_t curand_;

		CURAND_CHECK(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_MT19937));
		
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_, clock()));
		CURAND_CHECK(curandGenerateNormalDouble(curand_, data, count, mean, stddev));

		CURAND_CHECK(curandDestroyGenerator(curand_));
	}

	template<>
	void Uniform<float>(size_t count, float *data, float range)
	{
		curandGenerator_t curand_;
		cublasHandle_t cublas_;
		
		CUBLAS_CHECK(cublasCreate(&cublas_));
		CURAND_CHECK(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_MT19937));

		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_, clock()));
		CURAND_CHECK(curandGenerateUniform(curand_, data, count));
		float temp = 2 * range;
		surfing_gpu_scal(cublas_, count, &temp, data, 1);
		surfing_gpu_add_scalar(count, -range, data);

		CURAND_CHECK(curandDestroyGenerator(curand_));
		CUBLAS_CHECK(cublasDestroy(cublas_));
	}
	template<>
	void Uniform<double>(size_t count, double *data, double range)
	{
		curandGenerator_t curand_;
		cublasHandle_t cublas_;

		CUBLAS_CHECK(cublasCreate(&cublas_));
		CURAND_CHECK(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_MT19937));

		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_, clock()));
		CURAND_CHECK(curandGenerateUniformDouble(curand_, data, count));
		double temp = 2 * range;
		surfing_gpu_scal(cublas_, count, &temp, data, 1);
		surfing_gpu_add_scalar(count, -range, data);

		CURAND_CHECK(curandDestroyGenerator(curand_));
		CUBLAS_CHECK(cublasDestroy(cublas_));
	}

	void Integer(size_t count, unsigned int *data, unsigned int range)
	{
		std::mt19937_64 generator(clock());
		std::uniform_int_distribution<int> distribution(0, range-1);

		for (int i = 0; i< count; ++i) {
			data[i] = distribution(generator);
		}
	}
}
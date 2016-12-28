#include "test/RNG_test.h"
#include "basic/random_variable.h"
#include "basic/blob.h"
#include "basic/math_function.h"


using namespace surfing;

void Gaussian_TEST()
{
	Blob<float> param(100, 10, 5, 5);
	Gaussian<float>(param.count(), param.mutable_cpu_data(), 0, 0.01);

	cublasHandle_t cublas_;
	cublasCreate(&cublas_);
	float result;
	surfing_gpu_asum<float>(cublas_, param.count(), param.gpu_data(), 1, &result);
	LOG(INFO) << result;

	surfing_gpu_nrm2<float>(cublas_, param.count(), param.gpu_data(), 1, &result);
	LOG(INFO) << result;

}

void Uniform_TEST()
{
	Blob<float> param(100, 10, 5, 5);
	Uniform<float>(param.count(), param.mutable_cpu_data(),0.5);

	cublasHandle_t cublas_;
	cublasCreate(&cublas_);
	float result;
	surfing_gpu_asum<float>(cublas_, param.count(), param.gpu_data(), 1, &result);
	LOG(INFO) << result;

	surfing_gpu_nrm2<float>(cublas_, param.count(), param.gpu_data(), 1, &result);
	LOG(INFO) << result;

}

void Integer_TEST()
{
	unsigned int *b = new unsigned int[1000];
	Integer(1000, b, 1271000);
}


#include "basic/math_function.h"
#include "device_launch_parameters.h"

namespace surfing
{
	void surfing_gpu_memcpy(const size_t N, const void* X, void* Y)
	{
		if (X != Y)
		{
			CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
		}
	}

	template<>
	void surfing_gpu_asum<float>(cublasHandle_t handle, int N, const float *X, int incx, float *result)
	{
		CUBLAS_CHECK(cublasSasum(handle, N, X, incx, result));
	}
	template<>
	void surfing_gpu_asum<double>(cublasHandle_t handle, int N, const double *X, int incx, double *result)
	{
		CUBLAS_CHECK(cublasDasum(handle, N, X, incx, result));
	}

	template<>
	void surfing_gpu_nrm2<float>(cublasHandle_t handle, int N, const float *X, int incx, float *result)
	{
		CUBLAS_CHECK(cublasSnrm2(handle, N, X, incx, result));
	}
	template<>
	void surfing_gpu_nrm2<double>(cublasHandle_t handle, int N, const double *X, int incx, double *result)
	{
		CUBLAS_CHECK(cublasDnrm2(handle, N, X, incx, result));
	}

	template<>
	void surfing_gpu_dot<float>(cublasHandle_t handle, int N, const float *X, int incx, const float* Y, int incy, float *result)
	{
		CUBLAS_CHECK(cublasSdot(handle, N, X, incx,Y,incy,result));
	}
	template<>
	void surfing_gpu_dot<double>(cublasHandle_t handle, int N, const double *X, int incx, const double* Y, int incy, double *result)
	{
		CUBLAS_CHECK(cublasDdot(handle, N, X, incx, Y, incy, result));
	}


	template<>
	void surfing_gpu_gemm<float>(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k,
		const float *alpha, const float *A, int lda,
		const float *B, int ldb,
		const float *beta, float *C, int ldc)
	{

		CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));

	}
	template<>
	void surfing_gpu_gemm<double>(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k,
		const double *alpha, const double *A, int lda,
		const double *B, int ldb,
		const double *beta, double *C, int ldc)
	{
		CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
	}

	template<>
	void surfing_gpu_geam<float>(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n,
		const float *alpha, const float *A, int lda,
		const float *beta, const float *B, int ldb,
		float *C, int ldc)
	{
		CUBLAS_CHECK(cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
	}
	template<>
	void surfing_gpu_geam<double>(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n,
		const double *alpha, const double *A, int lda,
		const double *beta, const double *B, int ldb,
		double *C, int ldc)
	{
		CUBLAS_CHECK(cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
	}


	template <>
	void surfing_gpu_gemv<float>(cublasHandle_t handle, cublasOperation_t trans,
		int m, int n, const float* alpha,
		const float* A, int lda,
		const float* x, int incx,
		const float* beta, float* y, int incy)
	{
		CUBLAS_CHECK(cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
	}
	template <>
	void surfing_gpu_gemv<double>(cublasHandle_t handle, cublasOperation_t trans,
		int m, int n, const double* alpha,
		const double* A, int lda,
		const double* x, int incx,
		const double* beta, double* y, int incy)
	{
		CUBLAS_CHECK(cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
	}

	template<>
	void surfing_gpu_max<float>(cublasHandle_t handle, int n, const float* X, int incx, int* result)
	{
		CUBLAS_CHECK(cublasIsamax(handle, n, X, incx, result));
	}
	template<>
	void surfing_gpu_max<double>(cublasHandle_t handle, int n, const double * X, int incx, int* result)
	{
		CUBLAS_CHECK(cublasIdamax(handle, n, X, incx, result));
	}

	template<>
	void surfing_gpu_axpy<float>(cublasHandle_t handle, int N, const float* alpha,
		const float* X, int incx, float* Y, int incy)
	{
		CUBLAS_CHECK(cublasSaxpy(handle, N, alpha, X, incx, Y, incy));
	}
	template<>
	void surfing_gpu_axpy<double>(cublasHandle_t handle, int N, const double* alpha,
		const double* X, int incx, double* Y, int incy)
	{
		CUBLAS_CHECK(cublasDaxpy(handle, N, alpha, X, incx, Y, incy));
	}

	template<>
	void surfing_gpu_scal<float>(cublasHandle_t handle, int N, const float* alpha, float* X, int incx)
	{
		CUBLAS_CHECK(cublasSscal(handle, N, alpha, X, incx));
	}
	template<>
	void surfing_gpu_scal<double>(cublasHandle_t handle, int N, const double* alpha, double* X, int incx)
	{
		CUBLAS_CHECK(cublasDscal(handle, N, alpha, X, incx));
	}

	template<>
	void surfing_gpu_axpby<float>(cublasHandle_t handle, int N, const float* alpha, const float* X, const float* beta, float* Y)
	{
		surfing_gpu_scal<float>(handle, N, beta, Y, 1);
		surfing_gpu_axpy<float>(handle, N, alpha, X, 1, Y, 1);
	}
	template<>
	void surfing_gpu_axpby <double>(cublasHandle_t handle, int N, const double* alpha, const double* X, const double* beta, double* Y)
	{
		surfing_gpu_scal<double>(handle, N, beta, Y, 1);
		surfing_gpu_axpy<double>(handle, N, alpha, X, 1, Y, 1);
	}

	template <typename Dtype>
	__global__ void set_kernel(const int N, const Dtype alpha, Dtype* X)
	{
		CUDA_KERNEL_LOOP(index, N)
		{
			X[index] = alpha;
		}
	}
	template <> 
	void surfing_gpu_set<float>(const int N, const float alpha, float *X)
	{
		set_kernel<float> <<<SURFING_GET_BLOCK(N), SURFING_CUDA_NUM_THREADS >>>(N, alpha, X);
	}
	template <> 
	void surfing_gpu_set(const int N, const double alpha, double *X)
	{
		set_kernel<double> <<<SURFING_GET_BLOCK(N), SURFING_CUDA_NUM_THREADS >>>(N, alpha, X);
	}

	template <typename Dtype>
	__global__ void add_scalar_kernel(const int N, const Dtype alpha, Dtype* X)
	{
		CUDA_KERNEL_LOOP(index, N)
		{
			X[index] += alpha;
		}
	}
	template <>
	void surfing_gpu_add_scalar<float>(const int N, const float alpha, float *X)
	{
		add_scalar_kernel<float> <<<SURFING_GET_BLOCK(N), SURFING_CUDA_NUM_THREADS >>>(N, alpha, X);
	}
	template <>
	void surfing_gpu_add_scalar<double>(const int N, const double alpha, double *X)
	{
		add_scalar_kernel<double> <<<SURFING_GET_BLOCK(N), SURFING_CUDA_NUM_THREADS >>>(N, alpha, X);
	}

	__global__ void rounding_kernel(const int N, unsigned int range, unsigned int * X)
	{
		CUDA_KERNEL_LOOP(index, N)
		{
			X[index] %= range;
		}
	}
	void surfing_gpu_rounding(const int N, unsigned int range, unsigned int * X)
	{
		rounding_kernel<<<SURFING_GET_BLOCK(N), SURFING_CUDA_NUM_THREADS >>>(N, range, X);
	}
}


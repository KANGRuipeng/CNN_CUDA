#ifndef MATH_FUNCTION_H
#define MATH_FUNCTION_H

#include "basic/common.h"

namespace surfing
{
	/* Be careful that this two function just can set byte of chars , not other type*/
	inline void surfing_memset(const int N, const int alpha, void* X) { memset(X, alpha, N); }
	inline void surfing_gpu_memset(const size_t N, const int alpha, void* X) { CUDA_CHECK(cudaMemset(X, alpha, N)); };
	
	/* Function's defined in math_function.cpp */
	/*This function can set float our double data*/
	template <typename Dtype>
	void surfing_set(const int N, const Dtype alpha, Dtype* y);

	/* Function's defined in math_function.cu */
	void surfing_gpu_memcpy(const size_t N, const void* X, void* Y);

	template <typename Dtype>
	void surfing_gpu_asum(cublasHandle_t handle, int N, const Dtype *X, int incx, Dtype *result);

	template <typename Dtype>
	void surfing_gpu_nrm2(cublasHandle_t handle, int N, const Dtype *X, int incx, Dtype *result);

	template <typename Dtype>
	void surfing_gpu_dot(cublasHandle_t handle, int N, const Dtype *X, int incx, const Dtype* Y, int incy, Dtype *result);

	template <typename Dtype>
	void surfing_gpu_gemm(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k,
		const Dtype *alpha, const Dtype *A, int lda,
		const Dtype *B, int ldb,
		const Dtype *beta, Dtype *C, int ldc);

	template <typename Dtype>
	void surfing_gpu_geam(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n,
		const Dtype *alpha, const Dtype *A, int lda,
		const Dtype *beta, const Dtype *B, int ldb,
		Dtype *C, int ldc);

	template <typename Dtype>
	void surfing_gpu_gemv(cublasHandle_t handle, cublasOperation_t trans,
		int m, int n, const Dtype* alpha,
		const Dtype* A, int lda,
		const Dtype* x, int incx,
		const Dtype* beta, Dtype* y, int incy);

	template <typename Dtype>
	void surfing_gpu_max(cublasHandle_t handle, int n, const Dtype* X, int incx, int* result);

	template <typename Dtype>
	void surfing_gpu_axpy(cublasHandle_t handle, int N, const Dtype* alpha,
		const Dtype* X, int incx, Dtype* Y, int incy);
	
	template <typename Dtype>
	void surfing_gpu_scal(cublasHandle_t handle, int N, const Dtype* alpha, Dtype* X, int incx);

	template <typename Dtype>
	void surfing_gpu_axpby(cublasHandle_t handle, int N, const Dtype* alpha, const Dtype* X, const Dtype* beta, Dtype* Y);

	template <typename Dtype>
	void surfing_gpu_set(const int N, const Dtype alpha, Dtype* X);

	template <typename Dtype>
	void surfing_gpu_add_scalar(const int N, const Dtype alpha, Dtype* X);
	
	void surfing_gpu_rounding(const int N, unsigned int range, unsigned int * X);
}

#endif
#ifndef COMMON_H
#define COMMON_H

#include <cublas_v2.h>
#include <curand.h>

/*Add this, I just have to include common.h*/
#include "glog/logging.h"

#define CIFAR10

#define DISABLE_COPY_AND_ASSIGN(classname) \
	private:\
	classname(const classname&);\
	classname& operator=(const classname&)

#define INSTANTIATE_CLASS(classname)\
	template class classname<float>;\
	template class classname<double>

#define CUDA_CHECK(condition) \
	do{\
	cudaError_t error = condition;\
	CHECK_EQ(error,cudaSuccess)<< " " <<cudaGetErrorString(error);\
				}while(0)

#define CUDNN_CHECK(condition)\
	do{\
	cudnnStatus_t status = condition;\
	CHECK_EQ(status,CUDNN_STATUS_SUCCESS) << " "\
		<<cudnnGetErrorString(status);\
		} while (0)

#define CUBLAS_CHECK(condition)\
	do{\
	cublasStatus_t status = condition;\
	CHECK_EQ(status,CUBLAS_STATUS_SUCCESS) << " "\
	<<surfing::cublasGetErrorString(status);\
				} while (0)

#define CURAND_CHECK(condition)\
	do{\
	curandStatus_t status = condition;\
	CHECK_EQ(status,CURAND_STATUS_SUCCESS) << " "\
	<<surfing::curandGetErrorString(status);\
				}while(0)

#define CUDA_KERNEL_LOOP(i,n) \
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

namespace surfing
{
	const char* cublasGetErrorString(cublasStatus_t error);
	const char* curandGetErrorString(curandStatus_t error);

	const int SURFING_CUDA_NUM_THREADS = 512;
	inline int SURFING_GET_BLOCK(const int N) 
	{
		return (N + SURFING_CUDA_NUM_THREADS - 1) / SURFING_CUDA_NUM_THREADS;
	}
}

#endif
#include <iostream>
#include <cudnn.h>
#include <vector>

#include "basic/blob.h"
#include "basic/common.h"
#include "basic/cudnn_api.h"
#include "basic/math_function.h"

#include "test/batch_normalization_test.h"

using namespace surfing;

void Batch_Normalization_TEST()
{
	float A[9] = { 1, 2, 3, 2, 5, 4, -2, 8, 2 };

	float B[9] = { 0.1, 0.2, -0.2, 0.3, 0.5, -0.2, -0.2, 0.3, 0.7};

	Blob<float>* top;
	Blob<float>* bottom;

	bottom = new Blob<float>;
	bottom->Reshape(3, 3, 1, 1);
	bottom->set_cpu_data(A);

	top = new Blob<float>;
	top->Reshape(3, 3, 1, 1);
	top->set_cpu_diff(B);

	cudnnHandle_t handle_;
	cudnnTensorDescriptor_t bottom_desc_, top_desc_, bnScaleBiasMeanVarDesc_;

	float *bnScale_, *bnBias_;
	float *resultRunningMean_, *resultRunningVariance_;
	float *resultSaveMean_, *resultSaveVariance_;
	double exponentialAverageFactor_, epsilon_;
	float *resultBnScaleDiff_, *resultBnBiasDiff_;

	float *bnScale_cpu, *bnBias_cpu;
	float *resultRunningMean_cpu, *resultRunningVariance_cpu;
	float *resultSaveMean_cpu, *resultSaveVariance_cpu;
	float *resultBnScaleDiff_cpu, *resultBnBiasDiff_cpu;

	cudnnCreate(&handle_);
	cudnnCreateTensorDescriptor(&bottom_desc_);
	cudnnCreateTensorDescriptor(&top_desc_);
	cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc_);

	vector<int> shape;
	shape = bottom->shape();
	cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
		shape[0], shape[1], shape[2], shape[3]);

	shape = top->shape();
	cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
		shape[0], shape[1], shape[2], shape[3]);

	cudaMalloc(&bnScale_, sizeof(float) * shape[1]);
	cudaMalloc(&bnBias_, sizeof(float) * shape[1]);
	surfing_gpu_set(shape[1], (float)1.0, bnScale_);

	cudaMalloc(&resultBnScaleDiff_, sizeof(float) * shape[1]);
	cudaMalloc(&resultBnBiasDiff_, sizeof(float) * shape[1]);

	cudaMalloc(&resultRunningMean_, sizeof(float) * shape[1]);
	cudaMalloc(&resultRunningVariance_, sizeof(float) * shape[1]);

	cudaMalloc(&resultSaveMean_, sizeof(float) * shape[1]);
	cudaMalloc(&resultSaveVariance_, sizeof(float) * shape[1]);

	bnScale_cpu = new float[shape[1]];
	bnBias_cpu = new float[shape[1]];
	resultRunningMean_cpu = new float[shape[1]];
	resultRunningVariance_cpu = new float[shape[1]];
	resultSaveMean_cpu = new float[shape[1]];
	resultSaveVariance_cpu = new float[shape[1]];
	resultBnScaleDiff_cpu = new float[shape[1]];
	resultBnBiasDiff_cpu = new float[shape[1]];

	CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc_, bottom_desc_, CUDNN_BATCHNORM_SPATIAL));

	CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_, CUDNN_BATCHNORM_SPATIAL,
		cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
		bottom_desc_, bottom->gpu_data(),
		top_desc_, top->mutable_gpu_data(),
		bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
		1.0, resultRunningMean_, resultRunningVariance_,
		CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveVariance_));

	for (int i = 0; i < top->count(); i++)
	{
		LOG(INFO) << " top  data " << i << " " << top->cpu_data()[i];
	}


	surfing_gpu_memcpy(sizeof(float) * shape[1], resultRunningMean_, resultRunningMean_cpu);
	surfing_gpu_memcpy(sizeof(float) * shape[1], resultRunningVariance_, resultRunningVariance_cpu);
	surfing_gpu_memcpy(sizeof(float) * shape[1], resultSaveMean_, resultSaveMean_cpu);
	surfing_gpu_memcpy(sizeof(float) * shape[1], resultSaveVariance_, resultSaveVariance_cpu);

	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultRunningMean_ " << j << " " << resultRunningMean_cpu[j];
	}
	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultRunningVariance_ " << j << " " << resultRunningVariance_cpu[j];
	}
	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultSaveMean_ " << j << " " << resultSaveMean_cpu[j];
	}
	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultSaveVariance_ " << j << " " << resultSaveVariance_cpu[j];
	}

	CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_, CUDNN_BATCHNORM_SPATIAL,
		cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
		cudnn::dataType<float>::one, cudnn::dataType<float>::zero,
		bottom_desc_, bottom->gpu_data(),
		top_desc_, top->gpu_diff(),
		bottom_desc_, bottom->mutable_gpu_diff(),
		bnScaleBiasMeanVarDesc_, bnScale_, 
		resultBnScaleDiff_, resultBnBiasDiff_,
		CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveVariance_));

	for (int i = 0; i < bottom->count(); i++)
	{
		LOG(INFO) << " bottom diff " << i  << " "<< bottom->cpu_diff()[i];
	}

	surfing_gpu_memcpy(sizeof(float) * shape[1], resultBnScaleDiff_, resultBnScaleDiff_cpu);
	surfing_gpu_memcpy(sizeof(float) * shape[1], resultBnBiasDiff_, resultBnBiasDiff_cpu);

	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultBnScaleDiff_ " << j << " " << resultBnScaleDiff_cpu[j];
	}
	for (int j = 0; j < shape[1]; j++)
	{
		LOG(INFO) << " resultBnBiasDiff_ " << j << " " << resultBnBiasDiff_cpu[j];
	}

	cudnnDestroy(handle_);
	cudnnDestroyTensorDescriptor(bottom_desc_);
	cudnnDestroyTensorDescriptor(top_desc_);
	cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc_);

}

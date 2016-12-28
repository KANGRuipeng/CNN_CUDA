#include <iostream>
#include <cudnn.h>
#include <vector>

#include "basic/blob.h"
#include "basic/common.h"
#include "basic/cudnn_api.h"

using namespace std;
using namespace surfing;


void Pooling_TEST()
{
	float A[32] = { 1, 2, 3, 4, 5, 6, 7, 8, 9,
		1, 4, 7, 2, 5, 8, 3, 6, 9,
		2, 2, 3, 4, 6, 6, 7, 8, 10,
		1, -2, -4, 2, 1 };

	float B[8] = { 2, 4, 5, 6, 5, 6, 7, 8 };


	float C[32] = { 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0 };

	Blob<float>* top;
	Blob<float>* bottom;

	bottom = new Blob<float>;
	bottom->Reshape(1, 2, 4, 4);
	bottom->set_cpu_data(A);

	top = new Blob<float>;
	top->Reshape(1, 2, 2, 2);
	top->set_cpu_diff(B);

	cudnnHandle_t handle_;

	cudnnTensorDescriptor_t bottom_desc_, top_desc_;
	cudnnPoolingDescriptor_t pooling_desc_;
	cudnnPoolingMode_t mode_;

	cudnnCreate(&handle_);
	cudnnCreateTensorDescriptor(&bottom_desc_);
	cudnnCreateTensorDescriptor(&top_desc_);
	cudnnCreatePoolingDescriptor(&pooling_desc_);
	cudnnSetPooling2dDescriptor_v4(pooling_desc_, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
		2, 2, 0, 0, 2, 2);


	vector<int> shape;
	shape = bottom->shape();
	cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
		shape[0], shape[1], shape[2], shape[3]);

	shape = top->shape();
	cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
		shape[0], shape[1], shape[2], shape[3]);


	cudnnPoolingForward(handle_, pooling_desc_,
		cudnn::dataType<float>::one,
		bottom_desc_, bottom->gpu_data(), cudnn::dataType<float>::zero,
		top_desc_, top->mutable_gpu_data());
	
	for (int i = 0; i < top->count(); i++)
	{
		LOG(INFO) << top->cpu_data()[i];
	}
	bottom->set_cpu_data(C);

	cudnnPoolingBackward(handle_, pooling_desc_,
		cudnn::dataType<float>::one,
		top_desc_, top->gpu_data(),
		top_desc_, top->gpu_diff(),
		bottom_desc_, bottom->gpu_data(),
		cudnn::dataType<float>::zero,
		bottom_desc_, bottom->mutable_gpu_diff());

	for (int i = 0; i < bottom->count(); i++)
	{
		LOG(INFO) << bottom->cpu_diff()[i];
	}

	cudnnDestroy(handle_);
	cudnnDestroyTensorDescriptor(bottom_desc_);
	cudnnDestroyTensorDescriptor(top_desc_);
	cudnnDestroyPoolingDescriptor(pooling_desc_);
}
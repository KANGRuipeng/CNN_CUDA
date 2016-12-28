#include "layer/dropout_layer.h"

#include "basic/cudnn_api.h"
#include "basic/math_function.h"

#include <time.h>
#include <stdlib.h>


namespace surfing
{
	template <typename Dtype>
	DropoutLayer<Dtype>::DropoutLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreateDropoutDescriptor(&dropout_desc_);
	}

	template <typename Dtype>
	DropoutLayer<Dtype>::~DropoutLayer()
	{
		cudaFree(reserve);
		cudaFree(states);

		cudnnDestroy(handle_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyDropoutDescriptor(dropout_desc_);
	}

	template <typename Dtype>
	void DropoutLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape;

		shape = bottom[0]->shape();
		cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		shape = top->shape();
		cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);
		
		CUDNN_CHECK(cudnnDropoutGetStatesSize(handle_, &stateSizeInBytes));
		cudaMalloc(&states, stateSizeInBytes);

		CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(bottom_desc_, &reserveSize));
		cudaMalloc(&reserve, reserveSize);

		dropout_ratio_ = layer_param_.dropout_param().dropout_ratio();
	}

	template <typename Dtype>
	void DropoutLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		if (layer_param_.phase() == LayerParameter::TRAIN)
		{
			srand(clock());
			seed = 1 + (int)100 * rand() / (RAND_MAX + 1);

			CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_ratio_,
				states, stateSizeInBytes, seed));

			CUDNN_CHECK(cudnnDropoutForward(handle_, dropout_desc_,
				bottom_desc_, bottom[0]->gpu_data(),
				top_desc_, top->mutable_gpu_data(),
				reserve, reserveSize));
		}
		else
		{
			surfing_gpu_memcpy(sizeof(Dtype) * bottom[0]->count(), bottom[0]->gpu_data(), top->mutable_gpu_data());
		}

	}

	template <typename Dtype>
	void DropoutLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnDropoutBackward(handle_, dropout_desc_,
			top_desc_, top->gpu_diff(),
			bottom_desc_, bottom[0]->mutable_gpu_diff(),
			reserve, reserveSize));
	}

	INSTANTIATE_CLASS(DropoutLayer);
}
#include "layer/pooling_layer.h"

#include "basic/cudnn_api.h"

namespace surfing
{
	template <typename Dtype>
	PoolingLayer<Dtype>::PoolingLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreatePoolingDescriptor(&pooling_desc_);
	}

	template <typename Dtype>
	PoolingLayer<Dtype>::~PoolingLayer()
	{
		cudnnDestroy(handle_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyPoolingDescriptor(pooling_desc_);
	}

	template <typename Dtype>
	void PoolingLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape;
		shape = bottom[0]->shape();
		cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		shape = top->shape();
		cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		if (layer_param_.pooling_param().pooling_method() == PoolingParameter_PoolingMethod_AVE)
		{
			mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		}
		else if (layer_param_.pooling_param().pooling_method() == PoolingParameter_PoolingMethod_MAX)
		{
			mode_ = CUDNN_POOLING_MAX;
		}
		else
		{
			LOG(FATAL) << "Unknown pooling method !";
		}

		cudnnSetPooling2dDescriptor_v4(pooling_desc_, mode_, CUDNN_NOT_PROPAGATE_NAN,
			layer_param_.pooling_param().window_h(), layer_param_.pooling_param().window_w(),
			layer_param_.pooling_param().pad_h(), layer_param_.pooling_param().pad_w(),
			layer_param_.pooling_param().stride_h(), layer_param_.pooling_param().stride_w());

	}

	template <typename Dtype>
	void PoolingLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnPoolingForward(handle_, pooling_desc_,
			cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom[0]->gpu_data(), cudnn::dataType<Dtype>::zero,
			top_desc_, top->mutable_gpu_data()));
	}

	template <typename Dtype>
	void PoolingLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnPoolingBackward(handle_, pooling_desc_,
			cudnn::dataType<Dtype>::one,
			top_desc_, top->gpu_data(),
			top_desc_, top->gpu_diff(),
			bottom_desc_, bottom[0]->gpu_data(),
			cudnn::dataType<Dtype>::zero,
			bottom_desc_, bottom[0]->mutable_gpu_diff()));
	}

	INSTANTIATE_CLASS(PoolingLayer);
}
#include "layer/lrn_layer.h"

#include "basic/cudnn_api.h"

namespace surfing
{
	template <typename Dtype>
	LRNLayer<Dtype>::LRNLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreateLRNDescriptor(&lrn_desc_);
	}

	template <typename Dtype>
	LRNLayer<Dtype>::~LRNLayer()
	{
		cudnnDestroy(handle_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyLRNDescriptor(lrn_desc_);
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape;

		shape = bottom[0]->shape();
		cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		shape = top->shape();
		cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		CUDNN_CHECK(cudnnSetLRNDescriptor(lrn_desc_, layer_param_.lrn_param().local_size(), layer_param_.lrn_param().alpha(), 
			layer_param_.lrn_param().beta(), layer_param_.lrn_param().k()));
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnLRNCrossChannelForward(handle_, lrn_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			cudnn::dataType<Dtype>::one, bottom_desc_, bottom[0]->gpu_data(),
			cudnn::dataType<Dtype>::zero, top_desc_, top->mutable_gpu_data()));
	}

	template <typename Dtype>
	void LRNLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnLRNCrossChannelBackward(handle_, lrn_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			cudnn::dataType<Dtype>::one, top_desc_, top->gpu_data(),
			top_desc_, top->gpu_diff(),
			bottom_desc_, bottom[0]->gpu_data(),
			cudnn::dataType<Dtype>::zero, bottom_desc_, bottom[0]->mutable_gpu_diff()));
	}

	INSTANTIATE_CLASS(LRNLayer);
}
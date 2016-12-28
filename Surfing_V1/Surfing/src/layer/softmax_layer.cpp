#include "layer/softmax_layer.h"

#include "basic/cudnn_api.h"

namespace surfing
{
	template <typename Dtype>
	SoftmaxLayer<Dtype>::SoftmaxLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
	}

	template <typename Dtype>
	SoftmaxLayer<Dtype>::~SoftmaxLayer()
	{
		cudnnDestroy(handle_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
	}

	template <typename Dtype>
	void SoftmaxLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape;

		shape = bottom[0]->shape();
		cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);

		shape = top->shape();
		cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]);
	}

	template <typename Dtype>
	void SoftmaxLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
			cudnn::dataType<Dtype>::one, bottom_desc_, bottom[0]->gpu_data(),
			cudnn::dataType<Dtype>::zero, top_desc_, top->mutable_gpu_data()));
	}


	template <typename Dtype>
	void SoftmaxLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
			cudnn::dataType<Dtype>::one, top_desc_, top->gpu_data(),
			top_desc_, top->gpu_diff(),
			cudnn::dataType<Dtype>::zero, bottom_desc_, bottom[0]->mutable_gpu_diff()));
	}

	INSTANTIATE_CLASS(SoftmaxLayer);
}
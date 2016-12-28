#include "layer/tanh_layer.h"

#include "basic/cudnn_api.h"

namespace surfing
{
	template <typename Dtype>
	TanhLayer<Dtype>::TanhLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreateActivationDescriptor(&tanh_desc_);
		cudnnSetActivationDescriptor(tanh_desc_, CUDNN_ACTIVATION_RELU,
			CUDNN_NOT_PROPAGATE_NAN, 0.01);
	}

	template <typename Dtype>
	TanhLayer<Dtype>::~TanhLayer()
	{
		cudnnDestroy(handle_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyActivationDescriptor(tanh_desc_);
	}

	template <typename Dtype>
	void TanhLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
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
	void TanhLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnActivationForward(handle_, tanh_desc_,
			cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom[0]->gpu_data(), cudnn::dataType<Dtype>::zero,
			top_desc_, top->mutable_gpu_data()));
	}

	template <typename Dtype>
	void TanhLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnActivationBackward(handle_, tanh_desc_,
			cudnn::dataType<Dtype>::one,
			top_desc_, top->gpu_data(),
			top_desc_, top->gpu_diff(),
			bottom_desc_, bottom[0]->gpu_data(),
			cudnn::dataType<Dtype>::zero,
			bottom_desc_, bottom[0]->mutable_gpu_diff()));
	}

	INSTANTIATE_CLASS(TanhLayer);
}
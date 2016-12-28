#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H


#include <cudnn.h>

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class ConvolutionLayer : public Layer<Dtype>
	{
	public:
		ConvolutionLayer(const LayerParameter& param);
		~ConvolutionLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:
	private:
		cudnnHandle_t handle_[3];
		cudaStream_t stream_[3];

		cudnnTensorDescriptor_t bottom_desc_, top_desc_, bias_desc_;
		cudnnFilterDescriptor_t filter_desc_;
		cudnnConvolutionDescriptor_t conv_desc_;

		size_t workspace_size_[3];
		void* workspace_[3];
	};
}
#endif


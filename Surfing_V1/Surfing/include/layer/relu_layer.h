#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <cudnn.h>

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class ReluLayer : public Layer<Dtype>
	{
	public:
		ReluLayer(const LayerParameter& param);
		~ReluLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:

	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnActivationDescriptor_t relu_desc_;
	};
}
#endif
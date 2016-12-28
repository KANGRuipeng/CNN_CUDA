#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class SigmoidLayer : public Layer<Dtype>
	{
	public:
		SigmoidLayer(const LayerParameter& param);
		~SigmoidLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnActivationDescriptor_t sigmoid_desc_;
	};

}
#endif
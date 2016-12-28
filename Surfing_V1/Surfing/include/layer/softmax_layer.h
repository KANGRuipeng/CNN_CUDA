#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class SoftmaxLayer : public Layer<Dtype>
	{
	public:
		SoftmaxLayer(const LayerParameter& param);
		~SoftmaxLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:

	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
	};

}
#endif
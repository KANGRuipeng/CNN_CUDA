#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include <cudnn.h>

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class TanhLayer : public Layer<Dtype>
	{
	public:
		TanhLayer(const LayerParameter& param);
		~TanhLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:
	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnActivationDescriptor_t tanh_desc_;
	};
}
#endif
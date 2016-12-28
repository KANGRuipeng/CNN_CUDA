#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class PoolingLayer : public Layer<Dtype>
	{
	public:
		PoolingLayer(const LayerParameter& param);
		~PoolingLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnPoolingDescriptor_t pooling_desc_;	
		cudnnPoolingMode_t mode_;

	};
}
#endif
#ifndef LRN_LAYER_H
#define LRN_LAYER_H

#include <cudnn.h>

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class LRNLayer : public Layer<Dtype>
	{
	public:
		LRNLayer(const LayerParameter& param);
		~LRNLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:

	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnLRNDescriptor_t lrn_desc_;
	};
}
#endif
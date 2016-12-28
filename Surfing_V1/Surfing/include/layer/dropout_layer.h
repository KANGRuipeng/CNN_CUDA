#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include <cudnn.h>

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class DropoutLayer : public Layer<Dtype>
	{
	public:
		DropoutLayer(const LayerParameter& param);
		~DropoutLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:

	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnDropoutDescriptor_t dropout_desc_;

		float dropout_ratio_;

		void* reserve;
		size_t reserveSize;
		void* states;
		size_t stateSizeInBytes;

		unsigned long long seed;
	};
}
#endif
#ifndef INNER_PRODUCT_LAYER_H
#define INNER_PRODUCT_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class InnerProductLayer : public Layer<Dtype>
	{
	public:
		InnerProductLayer(const LayerParameter& param);
		~InnerProductLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:

	private:
		Blob<Dtype> bias_multiplier_;

		cublasHandle_t cublas_;
		int M_, K_, N_;
		Dtype alpha;
		Dtype beta;
	};
}
#endif
#ifndef CONCATENATE_LAYER_H
#define CONCATENATE_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class ConcatenateLayer : public Layer<Dtype>
	{
	public:
		ConcatenateLayer(const LayerParameter& param);
		~ConcatenateLayer();

		void Forward(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top);

	protected:
	private:
		vector<int> offset;
		int offset_sum;
	};
}
#endif
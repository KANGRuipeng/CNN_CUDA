#include "layer/concatenate_layer.h"
#include "basic/common.h"
#include "basic/math_function.h"

namespace surfing
{
	template <typename Dtype>
	ConcatenateLayer<Dtype>::ConcatenateLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
	}

	template <typename Dtype>
	ConcatenateLayer<Dtype>::~ConcatenateLayer()
	{
	}

	template <typename Dtype>
	void ConcatenateLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top)
	{
		offset_sum = 0;
		for (int i = 0; i < bottoms.size(); i++)
		{
			offset.push_back(bottoms[i]->shape()[1] * bottoms[i]->shape()[2] * bottoms[i]->shape()[3]);
			offset_sum += offset[i];
		}
	}

	template <typename Dtype>
	void ConcatenateLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top)
	{
		Dtype* out = top->mutable_gpu_data();
		for (int k = 0; k < top->shape()[0]; k++)
		{
			int offset_temp = 0;
			for (int i = 0; i < bottoms.size(); i++)
			{
				surfing_gpu_memcpy(offset[i] * sizeof(Dtype), bottoms[i]->gpu_data() + k * offset[i], out + offset_sum * k + offset_temp);
				offset_temp += offset[i];
			}
		}		
	}

	template <typename Dtype>
	void ConcatenateLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottoms, Blob<Dtype>*& top)
	{
		for (int k = 0; k < top->shape()[0]; k++)
		{
			int offset_temp = 0;
			for (int i = 0; i < bottoms.size(); i++)
			{
				surfing_gpu_memcpy(offset[i] * sizeof(Dtype), top->gpu_diff() + offset_sum * k + offset_temp, bottoms[i]->mutable_gpu_diff() + k * offset[i]);
				offset_temp += offset[i];
			}
		}
	}

	INSTANTIATE_CLASS(ConcatenateLayer);
}
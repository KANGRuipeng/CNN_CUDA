#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	Layer<Dtype>::Layer(const LayerParameter& param) : layer_param_(param)
	{
		LOG(INFO) << "Initial layer " << param.name();
	}

	template <typename Dtype>
	Layer<Dtype>::~Layer()
	{
		
	}

	INSTANTIATE_CLASS(Layer);
}
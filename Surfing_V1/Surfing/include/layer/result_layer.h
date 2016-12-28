#ifndef RESULT_LAYER_H
#define RESULT_LAYER_H

#include "layer/layer.h"

namespace surfing
{

	template <typename Dtype>
	class ResultLayer : public Layer<Dtype>
	{
	public:
		ResultLayer(const LayerParameter& param);
		~ResultLayer();

		void Reshape(Blob<Dtype>*& bottom);

		void Error_Calculate(Blob<Dtype>*& bottom, Blob<int>*& label);	
		void Accuracy(Blob<Dtype>*& bottom, Blob<int>*& label);

		void Set_Accurate_Count() { num_accurate_ = 0; }
		int Get_Accurate_Count() { return num_accurate_; }

		void Set_Loss() { loss_ = 0; }
		Dtype Get_Loss() { return loss_; }

	protected:
	private:
		cublasHandle_t cublas_;
		Dtype alpha;
		
		int num_accurate_;
		Dtype loss_;

		Dtype* diff;
		Dtype* temp_diff;
		int* temp_label;
		void* temp_gpu;
	};
}
#endif
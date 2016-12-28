#ifndef BATCH_NORMALIZATION_LAYER_H
#define BATCH_NORMALIZATION_LAYER_H

#include "layer/layer.h"

namespace surfing
{
	template <typename Dtype>
	class BatchNormalizationLayer : public Layer<Dtype>
	{
	public:
		BatchNormalizationLayer(const LayerParameter& param);
		~BatchNormalizationLayer();

		void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);
		void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top);

	protected:
	private:
		cudnnHandle_t handle_;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_, bnScaleBiasMeanVarDesc_;
		cublasHandle_t cublas_;

		Dtype *bnScale_, *bnBias_;
		Dtype *resultRunningMean_, *resultRunningVariance_;
		Dtype *resultSaveMean_, *resultSaveInvVariance_;
		double exponentialAverageFactor_;
		Dtype *resultBnScaleDiff_, *resultBnBiasDiff_;
		
		/* use this to count batch processed and calculate cumulative moving average */
		int batch_count_;

		int size_;

		Dtype alpha, beta;
	};
}
#endif
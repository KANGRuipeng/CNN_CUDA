#include "layer/batch_normalization_layer.h"
#include "basic/cudnn_api.h"
#include "basic/math_function.h"

namespace surfing
{
	template <typename Dtype>
	BatchNormalizationLayer<Dtype>::BatchNormalizationLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cudnnCreate(&handle_);
		cublasCreate(&cublas_);
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc_);
	}


	template <typename Dtype>
	BatchNormalizationLayer<Dtype>::~BatchNormalizationLayer()
	{
		cudnnDestroy(handle_);
		cublasDestroy(cublas_);
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc_);
	}

	template <typename Dtype>
	void BatchNormalizationLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		batch_count_ = 0;

		alpha = 1.0;
		beta = -1 * layer_param_.filter_learning_rate();

		vector<int> shape;
		shape = bottom[0]->shape();
		cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
			shape[0], shape[1], shape[2], shape[3]);

		shape = top->shape();
		cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<float>::type,
			shape[0], shape[1], shape[2], shape[3]);

		if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_CONV)
		{
			CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc_, bottom_desc_, CUDNN_BATCHNORM_SPATIAL));
			size_ = shape[1] * sizeof(Dtype);
		}
		else if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_IP)
		{
			CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc_, bottom_desc_, CUDNN_BATCHNORM_PER_ACTIVATION));
			size_ = shape[1] * shape[2] * shape[3] * sizeof(Dtype);
		}
		else
		{
			LOG(FATAL) << " Unknown type error ";
		}
		cudaMalloc(&bnScale_, size_);
		cudaMalloc(&bnBias_, size_);
		surfing_gpu_set(size_, (Dtype)1.0, bnScale_);

		cudaMalloc(&resultBnScaleDiff_, size_);
		cudaMalloc(&resultBnBiasDiff_, size_);

		cudaMalloc(&resultRunningMean_, size_);
		cudaMalloc(&resultRunningVariance_, size_);

		cudaMalloc(&resultSaveMean_, size_);
		cudaMalloc(&resultSaveInvVariance_, size_);
	}

	template <typename Dtype>
	void BatchNormalizationLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		if (layer_param_.phase() == LayerParameter::TRAIN)
		{
			batch_count_++;
			exponentialAverageFactor_ = 1.0 / (1.0 + batch_count_);

			if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_CONV)
			{
				CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_, CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
					bottom_desc_, bottom[0]->gpu_data(),
					top_desc_, top->mutable_gpu_data(),
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
					exponentialAverageFactor_, resultRunningMean_, resultRunningVariance_,
					CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));
			}
			else if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_IP)
			{
				CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_, CUDNN_BATCHNORM_PER_ACTIVATION,
					cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
					bottom_desc_, bottom[0]->gpu_data(),
					top_desc_, top->mutable_gpu_data(),
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
					exponentialAverageFactor_, resultRunningMean_, resultRunningVariance_,
					CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));
			}
			else
			{
				LOG(FATAL) << " Unknown type error ";
			}
		}
		else
		{
			if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_CONV)
			{
				CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_, CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
					bottom_desc_, bottom[0]->gpu_data(),
					top_desc_, top->mutable_gpu_data(),
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
					resultRunningMean_, resultRunningVariance_,
					CUDNN_BN_MIN_EPSILON));
			}
			else if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_IP)
			{
				CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_, CUDNN_BATCHNORM_PER_ACTIVATION,
					cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
					bottom_desc_, bottom[0]->gpu_data(),
					top_desc_, top->mutable_gpu_data(),
					bnScaleBiasMeanVarDesc_, bnScale_, bnBias_,
					resultRunningMean_, resultRunningVariance_,
					CUDNN_BN_MIN_EPSILON));
			}
			else
			{
				LOG(FATAL) << " Unknown type error ";
			}
		}
	}

	template <typename Dtype>
	void BatchNormalizationLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_CONV)
		{
			CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_, CUDNN_BATCHNORM_SPATIAL,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				bottom_desc_, bottom[0]->gpu_data(),
				top_desc_, top->gpu_diff(),
				bottom_desc_, bottom[0]->mutable_gpu_diff(),
				bnScaleBiasMeanVarDesc_, bnScale_,
				resultBnScaleDiff_, resultBnBiasDiff_,
				CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));
		}
		else if (layer_param_.batch_normalization_param().type() == BatchNormalizationParameter_Type_IP)
		{
			CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_, CUDNN_BATCHNORM_PER_ACTIVATION,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				bottom_desc_, bottom[0]->gpu_data(),
				top_desc_, top->gpu_diff(),
				bottom_desc_, bottom[0]->mutable_gpu_diff(),
				bnScaleBiasMeanVarDesc_, bnScale_,
				resultBnScaleDiff_, resultBnBiasDiff_,
				CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));
		}
		else
		{
			LOG(FATAL) << " Unknown type error ";
		}

		surfing_gpu_axpby(cublas_, size_ / sizeof(Dtype), &beta, resultBnScaleDiff_, &alpha, bnScale_);
		surfing_gpu_axpby(cublas_, size_ / sizeof(Dtype), &beta, resultBnBiasDiff_, &alpha, bnBias_);
	}
	INSTANTIATE_CLASS(BatchNormalizationLayer);
}
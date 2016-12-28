#include "layer/inner_product_layer.h"
#include "basic/math_function.h"

namespace surfing
{
	template <typename Dtype>
	InnerProductLayer<Dtype>::InnerProductLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cublasCreate(&cublas_);
		alpha = 1.0;
		beta = 0.0;
	}

	template <typename Dtype>
	InnerProductLayer<Dtype>::~InnerProductLayer()
	{
		cublasDestroy(cublas_);
	}

	template <typename Dtype>
	void InnerProductLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape = param_.shape();
		K_ = shape[1];
		N_ = shape[0];

		shape = bottom[0]->shape();
		M_ = shape[0];

		bias_multiplier_.Reshape(1, M_, 1, 1);
		surfing_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
	}

	template <typename Dtype>
	void InnerProductLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		surfing_gpu_gemm<Dtype>(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N_, M_, K_, &alpha,
			param_.gpu_data(), K_,
			bottom[0]->gpu_data(), K_,			
			&beta, top->mutable_gpu_data(), N_);

		surfing_gpu_gemm<Dtype>(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, N_, M_, 1, &alpha,
			bias_.gpu_data(), N_,
			bias_multiplier_.gpu_data(), 1,		
			&alpha, top->mutable_gpu_data(), N_);
	}

	template <typename Dtype>
	void InnerProductLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		surfing_gpu_gemv<Dtype>(cublas_, CUBLAS_OP_N, N_, M_,&alpha, 
			top->gpu_diff(), N_,
			bias_multiplier_.gpu_data(), 1,
			&beta, bias_.mutable_gpu_diff(), 1);

		surfing_gpu_gemm<Dtype>(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, N_, K_, M_, &alpha, 
			top->gpu_diff(), N_,
			bottom[0]->gpu_data(), K_,
			&beta, param_.mutable_gpu_diff(), N_);

		surfing_gpu_gemm<Dtype>(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, K_, M_, N_, &alpha, 
			param_.gpu_data(), K_,
			top->gpu_diff(), N_,
			&beta, bottom[0]->mutable_gpu_diff(), K_);		
	}

	INSTANTIATE_CLASS(InnerProductLayer);
}
#include "layer/result_layer.h"
#include "basic/math_function.h"

namespace surfing
{
	template <typename Dtype>
	ResultLayer<Dtype>::ResultLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		cublasCreate(&cublas_);		
		alpha = -1;
	}

	template <typename Dtype>
	ResultLayer<Dtype>::~ResultLayer()
	{
		delete[] temp_label;
		delete[] diff;
		delete[] temp_diff;
		cudaFree(temp_gpu);
		cublasDestroy(cublas_);
	}

#if defined(MNIST)

	template <typename Dtype>
	void ResultLayer<Dtype>::Reshape(Blob<Dtype>*& bottom)
	{
		vector<int> shape = bottom->shape();

		cudaMalloc(&temp_gpu, bottom->count() * sizeof(Dtype));
		diff = new Dtype[bottom->count()];
		temp_diff = new Dtype[bottom->count()];
		temp_label = new int[shape[0]];
		surfing_set(bottom->count(), (Dtype)0, temp_diff);		
	}

	template <typename Dtype>
	void ResultLayer<Dtype>::Error_Calculate(Blob<Dtype>*& bottom, Blob<int>*& label)
	{
		vector<int> shape = bottom->shape();
		memcpy(diff, temp_diff, sizeof(Dtype) * bottom->count());
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_set(1, (Dtype)1, diff + label->cpu_data()[i] + shape[1] * i);
		}
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), diff, temp_gpu);
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), bottom->gpu_data(), bottom->mutable_gpu_diff());
		surfing_gpu_axpy(cublas_, bottom->count(), &alpha, (Dtype*)temp_gpu, 1, bottom->mutable_gpu_diff(), 1);
	}

	template <typename Dtype>
	void ResultLayer<Dtype>::Accuracy(Blob<Dtype>*& bottom, Blob<int>*& label)
	{
		vector<int> shape = bottom->shape();
		memcpy(diff, temp_diff, sizeof(Dtype) * bottom->count());
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_set(1, (Dtype)1, diff + label->cpu_data()[i] + shape[1] * i);
		}
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), diff, temp_gpu);
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), bottom->gpu_data(), bottom->mutable_gpu_diff());
		surfing_gpu_axpy(cublas_, bottom->count(), &alpha, (Dtype*)temp_gpu, 1, bottom->mutable_gpu_diff(), 1);
		
		/* calculate accurate num */
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_gpu_max(cublas_, shape[1], bottom->gpu_data() + i*shape[1], 1, &temp_label[i]);
		}
		for (int i = 0; i < shape[0]; i++)
		{
			if ( (temp_label[i] - 1)== label->cpu_data()[i])
			{
				num_accurate_++;
			}
		}

		/* calculate loss*/
		Dtype loss_batch;
		if (layer_param_.reslut_param().losstype() == ResultParameter_LossType_L1)
		{		
			surfing_gpu_asum(cublas_, bottom->count(), bottom->gpu_diff(), 1, &loss_batch);
			loss_ += loss_batch;
		}
		else if (layer_param_.reslut_param().losstype() == ResultParameter_LossType_L2)
		{
			surfing_gpu_dot(cublas_, bottom->count(), bottom->gpu_diff(), 1, bottom->gpu_diff(), 1, &loss_batch);
			loss_ += loss_batch;
		}
		else
		{
			LOG(FATAL) << " Unknown loss type ";
		}
	}

#elif defined(CIFAR10)

	template <typename Dtype>
	void ResultLayer<Dtype>::Reshape(Blob<Dtype>*& bottom)
	{
		vector<int> shape = bottom->shape();

		cudaMalloc(&temp_gpu, bottom->count() * sizeof(Dtype));
		diff = new Dtype[bottom->count()];
		temp_diff = new Dtype[bottom->count()];
		temp_label = new int[shape[0]];
		surfing_set(bottom->count(), (Dtype)0, temp_diff);
	}

	template <typename Dtype>
	void ResultLayer<Dtype>::Error_Calculate(Blob<Dtype>*& bottom, Blob<int>*& label)
	{
		vector<int> shape = bottom->shape();
		memcpy(diff, temp_diff, sizeof(Dtype) * bottom->count());
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_set(1, (Dtype)1, diff + label->cpu_data()[i] + shape[1] * i);
		}
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), diff, temp_gpu);
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), bottom->gpu_data(), bottom->mutable_gpu_diff());
		surfing_gpu_axpy(cublas_, bottom->count(), &alpha, (Dtype*)temp_gpu, 1, bottom->mutable_gpu_diff(), 1);
	}

	template <typename Dtype>
	void ResultLayer<Dtype>::Accuracy(Blob<Dtype>*& bottom, Blob<int>*& label)
	{
		vector<int> shape = bottom->shape();
		memcpy(diff, temp_diff, sizeof(Dtype) * bottom->count());
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_set(1, (Dtype)1, diff + label->cpu_data()[i] + shape[1] * i);
		}
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), diff, temp_gpu);
		surfing_gpu_memcpy(bottom->count() * sizeof(Dtype), bottom->gpu_data(), bottom->mutable_gpu_diff());
		surfing_gpu_axpy(cublas_, bottom->count(), &alpha, (Dtype*)temp_gpu, 1, bottom->mutable_gpu_diff(), 1);

		/* calculate accurate num */
		for (int i = 0; i < shape[0]; i++)
		{
			surfing_gpu_max(cublas_, shape[1], bottom->gpu_data() + i*shape[1], 1, &temp_label[i]);
		}
		for (int i = 0; i < shape[0]; i++)
		{
			if ((temp_label[i] - 1) == label->cpu_data()[i])
			{
				num_accurate_++;
			}
		}

		/* calculate loss*/
		Dtype loss_batch;
		if (layer_param_.result_param().losstype() == ResultParameter_LossType_L1)
		{
			surfing_gpu_asum(cublas_, bottom->count(), bottom->gpu_diff(), 1, &loss_batch);
			loss_ += loss_batch;
		}
		else if (layer_param_.result_param().losstype() == ResultParameter_LossType_L2)
		{
			surfing_gpu_dot(cublas_, bottom->count(), bottom->gpu_diff(), 1, bottom->gpu_diff(), 1, &loss_batch);
			loss_ += loss_batch;
		}
		else
		{
			LOG(FATAL) << " Unknown loss type ";
		}
	}

#elif defined(IMAGENET)

#endif

	INSTANTIATE_CLASS(ResultLayer);
}
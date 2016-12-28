#include "layer/convolution_layer.h"
#include "basic/cudnn_api.h"
#include "basic/io_parameter.h"

namespace surfing
{
	template <typename Dtype>
	ConvolutionLayer<Dtype>::ConvolutionLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		for (int i = 0; i < 3; i++)
		{
			cudnnCreate(&handle_[i]);
			cudaStreamCreate(&stream_[i]);
			cudnnSetStream(handle_[i], stream_[i]);
		}
		cudnnCreateTensorDescriptor(&bottom_desc_);
		cudnnCreateTensorDescriptor(&top_desc_);
		cudnnCreateTensorDescriptor(&bias_desc_);
		cudnnCreateFilterDescriptor(&filter_desc_);
		cudnnCreateConvolutionDescriptor(&conv_desc_);
	}

	template <typename Dtype>
	ConvolutionLayer<Dtype>::~ConvolutionLayer()
	{
		for (int i = 0; i < 3; i++)
		{
			cudnnDestroy(handle_[i]);
			cudaStreamDestroy(stream_[i]);
			cudaFree(workspace_[i]);
		}
		cudnnDestroyTensorDescriptor(bottom_desc_);
		cudnnDestroyTensorDescriptor(top_desc_);
		cudnnDestroyTensorDescriptor(bias_desc_);
		cudnnDestroyFilterDescriptor(filter_desc_);
		cudnnDestroyConvolutionDescriptor(conv_desc_);		
	}

	template <typename Dtype>
	void ConvolutionLayer<Dtype>::Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		vector<int> shape;
		shape = param_.shape();
		CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, cudnn::dataType<Dtype>::type, CUDNN_TENSOR_NCHW,
			shape[0], shape[1], shape[2], shape[3]));
		/*For one output filter, there is a bias term*/
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			1, shape[0], 1, 1));

		shape = bottom[0]->shape();
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]));

		shape = top->shape();
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, cudnn::dataType<Dtype>::type,
			shape[0], shape[1], shape[2], shape[3]));

		int pad_h_ = layer_param_.conv_param().pad_h();
		int pad_w_ = layer_param_.conv_param().pad_w();
		int stride_h_ = layer_param_.conv_param().stride_h();
		int stride_w_ = layer_param_.conv_param().stride_w();

		CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, pad_h_, pad_w_, stride_h_, stride_w_, 1, 1, CUDNN_CONVOLUTION));

		CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0], bottom_desc_, filter_desc_,
			conv_desc_, top_desc_, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &workspace_size_[0]));
		cudaMalloc(&workspace_[0], workspace_size_[0]);
		
		CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[2], filter_desc_, top_desc_,
			conv_desc_, bottom_desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_size_[2]));
		cudaMalloc(&workspace_[1], workspace_size_[1]);

		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[1], bottom_desc_, top_desc_, 
			conv_desc_, filter_desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &workspace_size_[1]));
		cudaMalloc(&workspace_[2], workspace_size_[2]);
	}

	template <typename Dtype>
	void ConvolutionLayer<Dtype>::Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		//vector<int> t = bottom->shape();
		//LOG(INFO) <<t[0] << " " << t[1] << " " << t[2] << " " << t[3];
		//cv::Mat img(t[2],t[3], CV_8UC3);

		//for (int j = 0; j < t[2]; j++)
		//{
		//	for (int k = 0; k < t[2]; k++)
		//	{
		//		img.at<unsigned char>(j, k * 3) = (int)(bottom->cpu_data()[j * t[2] + k]*256);
		//		img.at<unsigned char>(j, k * 3 + 1) = (int)(bottom->cpu_data()[t[2] * t[2] + j * t[2] + k]*256);
		//		img.at<unsigned char>(j, k * 3 + 2) = (int)(bottom->cpu_data()[2 * t[2] * t[2] + j * t[2] + k]*256);
		//	}
		//}

		//cv::imshow("new", img);
		//cv::waitKey(0);

		//vector<int> t = bottom->shape();
		//LOG(INFO) <<t[0] << " " << t[1] << " " << t[2] << " " << t[3];
		//cv::Mat img(t[2],t[3], CV_8UC3);

		//for (int j = 0; j < t[2]; j++)
		//{
		//	for (int k = 0; k < t[2]; k++)
		//	{
		//		img.at<unsigned char>(j, k * 3) = (int)(bottom->cpu_data()[j * t[2] + k] * 256);
		//		img.at<unsigned char>(j, k * 3 + 1) = (int)(bottom->cpu_data()[j * t[2] + k] * 256);
		//		img.at<unsigned char>(j, k * 3 + 2) = (int)(bottom->cpu_data()[j * t[2] + k] * 256);
		//	}
		//}

		//cv::imshow("new", img);
		//cv::waitKey(0);

		CUDNN_CHECK(cudnnConvolutionForward(handle_[0], cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom[0]->gpu_data(),
			filter_desc_, param_.gpu_data(),
			conv_desc_, CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
			workspace_[0], workspace_size_[0],
			cudnn::dataType<Dtype>::zero,
			top_desc_, top->mutable_gpu_data()));

		CUDNN_CHECK(cudnnAddTensor(handle_[0], cudnn::dataType<Dtype>::one,
			bias_desc_, bias_.gpu_data(),
			cudnn::dataType<Dtype>::one,
			top_desc_, top->mutable_gpu_data()
			));
	}

	template <typename Dtype>
	void ConvolutionLayer<Dtype>::Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top)
	{
		CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[0], cudnn::dataType<Dtype>::one,
			filter_desc_, param_.gpu_data(),
			top_desc_, top->gpu_diff(),
			conv_desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
			workspace_[1], workspace_size_[1],
			cudnn::dataType<Dtype>::zero,
			bottom_desc_, bottom[0]->mutable_gpu_diff()));

		CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1], cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom[0]->gpu_data(),
			top_desc_, top->gpu_diff(),
			conv_desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
			workspace_[2], workspace_size_[2],
			cudnn::dataType<Dtype>::zero,
			filter_desc_, param_.mutable_gpu_diff()));

		CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[2], cudnn::dataType<Dtype>::one,
			top_desc_, top->gpu_diff(),
			cudnn::dataType<Dtype>::zero,
			bias_desc_, bias_.mutable_gpu_diff()));
	}

	INSTANTIATE_CLASS(ConvolutionLayer);
}



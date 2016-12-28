#ifndef LAYER_H
#define LAYER_H

#include <cudnn.h>

#include "basic/common.h"
#include "basic/blob.h"
#include "surfing.pb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace surfing
{
	template <typename Dtype>
	class Layer
	{
	public:
		explicit Layer(const LayerParameter& param);
		virtual ~Layer();

		inline LayerParameter& layer_param() { return layer_param_; }

		inline Blob<Dtype>& param() { return param_; }
		inline Blob<Dtype>& bias() { return bias_; }
		inline Blob<Dtype>& history_param() { return history_param_; }
		inline Blob<Dtype>& history_bias() { return history_bias_; }

		virtual void Forward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top) {}
		virtual void Backward(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top) {}
		virtual void Reshape(vector<Blob<Dtype>*>& bottom, Blob<Dtype>*& top) {}

	protected:
		LayerParameter layer_param_;
		Blob<Dtype> param_;
		Blob<Dtype> bias_;
		Blob<Dtype> history_param_;
		Blob<Dtype> history_bias_;

	private:
		DISABLE_COPY_AND_ASSIGN(Layer);
	};

}
#endif

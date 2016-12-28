#include "net.h"

#include <iostream>
#include <thread>

#include "basic/random_variable.h"
#include "basic/math_function.h"

#include "layer/batch_normalization_layer.h"
#include "layer/concatenate_layer.h"
#include "layer/convolution_layer.h"
#include "layer/inner_product_layer.h"
#include "layer/pooling_layer.h"
#include "layer/sigmoid_layer.h"
#include "layer/relu_layer.h"
#include "layer/softmax_layer.h"
#include "layer/tanh_layer.h"
#include "layer/dropout_layer.h"
#include "layer/lrn_layer.h"

namespace surfing
{
	template <typename Dtype>
	Net<Dtype>::Net(const string& param_file, bool pre_train)
	{
		ReadProtoFromTextFile(param_file.c_str(), &net_param_);
		Init();
		if (pre_train)
		{
			FromProto(param_file.c_str(), false);
		}
		Init_Desc();

		cublasCreate(&cublas_);
	}

	template <typename Dtype>
	Net<Dtype>::~Net()
	{
		cublasDestroy(cublas_);
	}

	/* Init data and result layer for train and test separately*/
	template <typename Dtype>
	void Net<Dtype>::Init()
	{
		/*Init layer*/
		for (int layer_id = 0; layer_id < net_param_.layer_size(); layer_id++)
		{
			const LayerParameter& layer_param = net_param_.layer(layer_id);
			/* This part is core ! Be careful to modify*/
			if ((layer_param.type() == LayerParameter::DATA) && (layer_param.phase() == LayerParameter::TRAIN))
			{
				for (int i = 0; i < layer_param.bottom().size(); i++)
				{
					blob_names_priority_.insert(make_pair(layer_param.bottom(i).data(), -1));
				}
				train_data_layer_ = new DataLayer<Dtype>(layer_param);
				for (int j = 0; j < layer_param.bottom().size(); j++)
				{
					train_input_names_id_.insert(make_pair(layer_param.bottom(j).data(), j));
				}
			}
			else if ((layer_param.type() == LayerParameter::DATA) && (layer_param.phase() == LayerParameter::TEST))
			{
				test_data_layer_ = new DataLayer<Dtype>(layer_param);
				for (int j = 0; j < layer_param.bottom().size(); j++)
				{
					test_input_names_id_.insert(make_pair(layer_param.bottom(j).data(), j));
				}
			}
			else if ((layer_param.type() == LayerParameter::RESULT) && (layer_param.phase() == LayerParameter::TRAIN))
			{
				train_result_layer_ = new ResultLayer<Dtype>(layer_param);
			}
			else if ((layer_param.type() == LayerParameter::RESULT) && (layer_param.phase() == LayerParameter::TEST))
			{
				test_result_layer_ = new ResultLayer<Dtype>(layer_param);
			}
			else
			{
				int max_priority = blob_names_priority_.find(layer_param.bottom(0).data())->second;
				for (int i = 0; i < layer_param.bottom().size(); i++)
				{
					if (blob_names_priority_.find(layer_param.bottom(i).data())->second > max_priority)
					{
						max_priority = blob_names_priority_.find(layer_param.bottom(i).data())->second;
					}
				}
				if (blob_names_priority_.find(layer_param.top().data()) != blob_names_priority_.end())
				{
					blob_names_priority_.erase(layer_param.top().data());
				}
				blob_names_priority_.insert(make_pair(layer_param.top().data(), max_priority + 1));

				DLOG(INFO) << layer_param.name() << " " << max_priority + 1;
				layer_names_priority_.insert(make_pair(layer_param.name(), max_priority + 1));

				layers_.push_back(Init_Layers(layer_param));
			}
		}
		LOG(INFO) << " General layer number " << layers_.size();

		/* used to save data shape and param shape, here this */
		vector<int> data_shape(4);
		data_shape[0] = train_data_layer_->Batch_Size();
		vector<int> param_shape(4);

		for (int layer_id = 0; layer_id < layers_.size(); layer_id++)
		{
			/*Get the data blob shape*/
			if (layers_[layer_id]->layer_param().has_channels()) { data_shape[1] = layers_[layer_id]->layer_param().channels(); }
			if (layers_[layer_id]->layer_param().has_height()) { data_shape[2] = layers_[layer_id]->layer_param().height(); }
			if (layers_[layer_id]->layer_param().has_width()) { data_shape[3] = layers_[layer_id]->layer_param().width(); }

			if (layers_[layer_id]->layer_param().type() == LayerParameter::CONVOLUTION)
			{
				data_shape[1] = layers_[layer_id]->layer_param().conv_param().num_output();

				param_shape[0] = layers_[layer_id]->layer_param().conv_param().num_output();
				param_shape[1] = layers_[layer_id]->layer_param().conv_param().filter_parameter().channels();
				param_shape[2] = param_shape[3] = layers_[layer_id]->layer_param().conv_param().kernel_size();

				Dtype bias = layers_[layer_id]->layer_param().conv_param().filter_parameter().bias();
				layers_[layer_id]->bias().Reshape(1, param_shape[1], 1, 1);
				layers_[layer_id]->history_bias().Reshape(1, param_shape[1], 1, 1);
				surfing_set(layers_[layer_id]->bias().count(), bias, layers_[layer_id]->bias().mutable_cpu_data());

				layers_[layer_id]->param().Reshape(param_shape);
				layers_[layer_id]->history_param().Reshape(param_shape);
				if (layers_[layer_id]->layer_param().conv_param().filter_parameter().filter() == FilterParameter::UNIFORM)
				{
					Dtype range = layers_[layer_id]->layer_param().conv_param().filter_parameter().range();
					Uniform<Dtype>(layers_[layer_id]->param().count(), layers_[layer_id]->param().mutable_gpu_data(), range);
				}
				else if (layers_[layer_id]->layer_param().conv_param().filter_parameter().filter() == FilterParameter::GAUSSIAN)
				{
					Dtype mean = layers_[layer_id]->layer_param().conv_param().filter_parameter().mean();
					Dtype std = layers_[layer_id]->layer_param().conv_param().filter_parameter().std();
					Gaussian<Dtype>(layers_[layer_id]->param().count(), layers_[layer_id]->param().mutable_gpu_data(), mean, std);
				}
				else
				{
					LOG(FATAL) << " Unknown filter type! ";
				}
				DLOG(INFO) << " Param " << param_shape[0] << " " << param_shape[1] << " " << param_shape[2] << " " << param_shape[3];
			}
			else if (layers_[layer_id]->layer_param().type() == LayerParameter::INNERPRODUCT)
			{
				data_shape[1] = layers_[layer_id]->layer_param().inner_product_param().num_output();

				param_shape[0] = layers_[layer_id]->layer_param().inner_product_param().num_output();
				param_shape[1] = layers_[layer_id]->layer_param().inner_product_param().filter_parameter().channels();
				param_shape[2] = param_shape[3] = 1;

				Dtype bias = layers_[layer_id]->layer_param().conv_param().filter_parameter().bias();
				layers_[layer_id]->bias().Reshape(1, param_shape[0], 1, 1);
				layers_[layer_id]->history_bias().Reshape(1, param_shape[0], 1, 1);
				surfing_set(layers_[layer_id]->bias().count(), bias, layers_[layer_id]->bias().mutable_cpu_data());

				layers_[layer_id]->param().Reshape(param_shape);
				layers_[layer_id]->history_param().Reshape(param_shape);
				if (layers_[layer_id]->layer_param().conv_param().filter_parameter().filter() == FilterParameter::UNIFORM)
				{
					Dtype range = layers_[layer_id]->layer_param().conv_param().filter_parameter().range();
					Uniform<Dtype>(layers_[layer_id]->param().count(), layers_[layer_id]->param().mutable_gpu_data(), range);
				}
				else if (layers_[layer_id]->layer_param().conv_param().filter_parameter().filter() == FilterParameter::GAUSSIAN)
				{
					Dtype mean = layers_[layer_id]->layer_param().conv_param().filter_parameter().mean();
					Dtype std = layers_[layer_id]->layer_param().conv_param().filter_parameter().std();
					Gaussian<Dtype>(layers_[layer_id]->param().count(), layers_[layer_id]->param().mutable_gpu_data(), mean, std);
				}
				else
				{
					LOG(FATAL) << " Unknown filter type! ";
				}
				DLOG(INFO) << " Param " << param_shape[0] << " " << param_shape[1] << " " << param_shape[2] << " " << param_shape[3];
			}
			else if (layers_[layer_id]->layer_param().type() == LayerParameter::SIGMOID ||
				layers_[layer_id]->layer_param().type() == LayerParameter::RELU ||
				layers_[layer_id]->layer_param().type() == LayerParameter::TANH ||
				layers_[layer_id]->layer_param().type() == LayerParameter::DROPOUT ||
				layers_[layer_id]->layer_param().type() == LayerParameter::POOLING ||
				layers_[layer_id]->layer_param().type() == LayerParameter::CONCATENATE ||
				layers_[layer_id]->layer_param().type() == LayerParameter::SOFTMAX ||
				layers_[layer_id]->layer_param().type() == LayerParameter::BATCHNORMALIZATION)
			{
				DLOG(INFO) << "Do Nothing! ";
			}
			else
			{
				LOG(FATAL) << " Unknow layer type ";
			}
			/*New a blob and get the ID name relationship*/
			if (blob_names_id_.find(layers_[layer_id]->layer_param().top().data()) == blob_names_id_.end())
			{
				int i = blobs_.size();
				DLOG(INFO) << " blob size " << i << "name " << layers_[layer_id]->layer_param().top().data();
				blob_names_id_.insert(make_pair(layers_[layer_id]->layer_param().top().data(), i));
				blobs_.push_back(new Blob<Dtype>(data_shape));
				DLOG(INFO) << data_shape[0] << " " << data_shape[1] << " " << data_shape[2] << " " << data_shape[3];
			}
		}
	}

	/* when add new layers, this function should be updated*/
	template <typename Dtype>
	Layer<Dtype>* Net<Dtype>::Init_Layers(const LayerParameter& param)
	{
		int i;
		i = layers_.size();
		layer_names_id_.insert(make_pair(param.name(), i));
		switch (param.type())
		{
		case LayerParameter::BATCHNORMALIZATION:
			return new BatchNormalizationLayer<Dtype>(param);
		case LayerParameter::CONCATENATE:
			return new ConcatenateLayer<Dtype>(param);
		case LayerParameter::CONVOLUTION:
			return new ConvolutionLayer<Dtype>(param);
		case LayerParameter::DROPOUT:
			return new DropoutLayer<Dtype>(param);
		case LayerParameter::INNERPRODUCT:
			return new InnerProductLayer<Dtype>(param);
		case LayerParameter::LRN:
			return new LRNLayer<Dtype>(param);
		case LayerParameter::POOLING:
			return new PoolingLayer<Dtype>(param);
		case LayerParameter::SIGMOID:
			return new SigmoidLayer<Dtype>(param);
		case LayerParameter::RELU:
			return new ReluLayer<Dtype>(param);
		case LayerParameter::TANH:
			return new TanhLayer<Dtype>(param);
		case LayerParameter::SOFTMAX:
			return new SoftmaxLayer<Dtype>(param);
		default:
			LOG(FATAL) << " Unknown Layer Type ";
			return NULL;
		}
	}

	template <typename Dtype>
	void Net<Dtype>::Init_Desc()
	{
		/*This part of code is used to build the 2-D array of priority*/
		map<string, int>::iterator it;
		int max = 0;
		for (it = layer_names_priority_.begin(); it != layer_names_priority_.end(); it++)
		{
			if (it->second > max) { max = it->second; }
		}
		layer_priority_.resize(max + 1);
		for (int i = 0; i <= max; i++)
		{
			for (it = layer_names_priority_.begin(); it != layer_names_priority_.end(); it++)
			{
				if (it->second == i)
				{
					layer_priority_[i].push_back(it->first);
				}
			}
		}

		train_bottom_blobs_.resize(layers_.size());
		test_bottom_blobs_.resize(layers_.size());
		/* Loop to build train blobs */
		for (int layer_id = 0; layer_id < layers_.size(); layer_id++)
		{
			for (int j = 0; j < layers_[layer_id]->layer_param().bottom_size(); j++)
			{
				if (blob_names_id_.find(layers_[layer_id]->layer_param().bottom(j)) != blob_names_id_.end())
				{
					train_bottom_blobs_[layer_id].push_back(blobs_[blob_names_id_.find(layers_[layer_id]->layer_param().bottom(j))->second]);
				}
				else
				{
					train_bottom_blobs_[layer_id].push_back(train_data_layer_->Data()[train_input_names_id_.find(layers_[layer_id]->layer_param().bottom(j))->second]);
				}
			}
			train_top_blobs_.push_back(blobs_[blob_names_id_.find(layers_[layer_id]->layer_param().top())->second]);
		}

		for (int layer_id = 0; layer_id < layers_.size(); layer_id++)
		{
			for (int j = 0; j < layers_[layer_id]->layer_param().bottom_size(); j++)
			{
				if (blob_names_id_.find(layers_[layer_id]->layer_param().bottom(j)) != blob_names_id_.end())
				{
					test_bottom_blobs_[layer_id].push_back(blobs_[blob_names_id_.find(layers_[layer_id]->layer_param().bottom(j))->second]);
				}
				else
				{
					test_bottom_blobs_[layer_id].push_back(test_data_layer_->Data()[train_input_names_id_.find(layers_[layer_id]->layer_param().bottom(j))->second]);
				}
			}
			test_top_blobs_.push_back(blobs_[blob_names_id_.find(layers_[layer_id]->layer_param().top())->second]);
		}


		for (int i = 0; i < layer_priority_.size(); i++)
		{
			for (int j = 0; j < layer_priority_[i].size(); j++)
			{
				int layer_id = layer_names_id_.find(layer_priority_[i][j])->second;
				DLOG(INFO) << " layer id " << layer_id;

				layers_[layer_id]->Reshape(train_bottom_blobs_[layer_id], train_top_blobs_[layer_id]);
			}
		}
		train_result_layer_->Reshape(blobs_[blobs_.size() - 1]);
		test_result_layer_->Reshape(blobs_[blobs_.size() - 1]);
	}

	template <typename Dtype>
	void Net<Dtype>::Train_Forward()
	{
		for (int i = 0; i < layer_priority_.size(); i++)
		{
			for (int j = 0; j < layer_priority_[i].size(); j++)
			{
				int layer_id = layer_names_id_.find(layer_priority_[i][j])->second;
				DLOG(INFO) << " layer id " << layer_id;
				layers_[layer_id]->layer_param().set_phase(LayerParameter::TRAIN);
				layers_[layer_id]->Forward(train_bottom_blobs_[layer_id], train_top_blobs_[layer_id]);
			}
		}
		//Should Update later 
		train_result_layer_->Error_Calculate(blobs_[blobs_.size() - 1], train_data_layer_->Label());
	}


	template <typename Dtype>
	void Net<Dtype>::Train_Backward()
	{
		for (int i = layer_priority_.size() - 1; i >= 0; i--)
		{
			for (int j = 0; j < layer_priority_[i].size(); j++)
			{
				int layer_id = layer_names_id_.find(layer_priority_[i][j])->second;
				DLOG(INFO) << " layer id " << layer_id;

				layers_[layer_id]->Backward(train_bottom_blobs_[layer_id], train_top_blobs_[layer_id]);
			}
		}
	}


	template <typename Dtype>
	void Net<Dtype>::Apply_Gradient(Dtype global_lr, Dtype momentum)
	{
		Dtype gamma = 1.0;
		for (int layer_id = 0; layer_id < layers_.size(); layer_id++)
		{
			if (layers_[layer_id]->layer_param().type() == LayerParameter::CONVOLUTION ||
				layers_[layer_id]->layer_param().type() == LayerParameter::INNERPRODUCT)
			{

				filter_lr_ = layers_[layer_id]->layer_param().filter_learning_rate() * global_lr;
				bias_lr_ = layers_[layer_id]->layer_param().bias_learning_rate() * global_lr;

				surfing_gpu_axpby<Dtype>(cublas_, layers_[layer_id]->param().count(), &filter_lr_, layers_[layer_id]->param().gpu_diff(),
					&momentum, layers_[layer_id]->history_param().mutable_gpu_diff());
				surfing_gpu_axpby<Dtype>(cublas_, layers_[layer_id]->bias().count(), &bias_lr_, layers_[layer_id]->bias().gpu_diff(),
					&momentum, layers_[layer_id]->history_bias().mutable_gpu_diff());

				surfing_gpu_axpy(cublas_, layers_[layer_id]->param().count(), &gamma,
					layers_[layer_id]->history_param().gpu_diff(), 1, layers_[layer_id]->param().mutable_gpu_data(), 1);
				surfing_gpu_axpy(cublas_, layers_[layer_id]->bias().count(), &gamma,
					layers_[layer_id]->history_bias().gpu_diff(), 1, layers_[layer_id]->bias().mutable_gpu_data(), 1);

				//surfing_gpu_axpy(cublas_, layers_[layer_id]->param().count(), &filter_lr_,
				//	layers_[layer_id]->param().gpu_diff(), 1, layers_[layer_id]->param().mutable_gpu_data(), 1);
				//surfing_gpu_axpy(cublas_, layers_[layer_id]->bias().count(), &bias_lr_,
				//	layers_[layer_id]->bias().gpu_diff(), 1, layers_[layer_id]->bias().mutable_gpu_data(), 1);
			}
			else
			{
				//LOG(INFO) << " No update ";
			}
		}
	}

	template <typename Dtype>
	void Net<Dtype>::Test_Forward()
	{
		
		for (int i = 0; i < layer_priority_.size(); i++)
		{
			for (int j = 0; j < layer_priority_[i].size(); j++)
			{
				int layer_id = layer_names_id_.find(layer_priority_[i][j])->second;
				DLOG(INFO) << " layer id " << layer_id;
				layers_[layer_id]->layer_param().set_phase(LayerParameter::TEST);
				layers_[layer_id]->Forward(test_bottom_blobs_[layer_id], test_top_blobs_[layer_id]);
			}
		}
		test_result_layer_->Accuracy(blobs_[blobs_.size() - 1], test_data_layer_->Label());
	}

	template <typename Dtype>
	void Net<Dtype>::ToProto(const char* filename, bool write_diff = false)
	{
		for (int layer_id = 0; layer_id < net_param_.layer_size(); layer_id++)
		{
			if (net_param_.layer(layer_id).type() == LayerParameter::CONVOLUTION
				|| net_param_.layer(layer_id).type() == LayerParameter::INNERPRODUCT)
			{
				if (net_param_.layer(layer_id).has_bias())
				{
					net_param_.mutable_layer(layer_id)->clear_bias();
				}
				BlobProto* bias = new BlobProto;
				layers_[layer_names_id_.find(net_param_.layer(layer_id).name())->second]->bias().ToProto(bias, write_diff);
				net_param_.mutable_layer(layer_id)->set_allocated_bias(bias);

				if (net_param_.layer(layer_id).has_param())
				{
					net_param_.mutable_layer(layer_id)->clear_param();
				}
				BlobProto* param = new BlobProto;
				layers_[layer_names_id_.find(net_param_.layer(layer_id).name())->second]->param().ToProto(param, write_diff);
				net_param_.mutable_layer(layer_id)->set_allocated_param(param);

				/* here I can't delete what I new, just because they are used to save data*/
			}
			else
			{
				//LOG(INFO) << " No Parameter to be saved !";
			}
		}
		WriteProtoToTextFile(net_param_, filename);
	}

	template <typename Dtype>
	void Net<Dtype>::FromProto(const char* filename, bool reshape = false)
	{
		for (int layer_id = 0; layer_id < net_param_.layer_size(); layer_id++)
		{
			if (net_param_.layer(layer_id).type() == LayerParameter::CONVOLUTION
				|| net_param_.layer(layer_id).type() == LayerParameter::INNERPRODUCT)
			{
				layers_[layer_names_id_.find(net_param_.layer(layer_id).name())->second]->bias().FromProto(net_param_.layer(layer_id).bias(), false);
				layers_[layer_names_id_.find(net_param_.layer(layer_id).name())->second]->param().FromProto(net_param_.layer(layer_id).param(), false);
			}

		}
	}


	INSTANTIATE_CLASS(Net);
}
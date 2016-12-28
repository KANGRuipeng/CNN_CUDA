#include "layer/data_layer.h"
#include "basic/random_variable.h"
#include "basic/math_function.h"

#include <time.h>
#include <stdlib.h>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace surfing
{
	template <typename Dtype>
	DataLayer<Dtype>::DataLayer(const LayerParameter& param) : Layer<Dtype>(param)
	{
		/* This part is used for init blobs, here is prepare for multi-scale input*/
		batch_size_ = layer_param_.data_param().batch_size();
		vector<int> shape(4);
		shape[0] = batch_size_;
		shape[1] = layer_param_.channels();
		for (int i = 0; i < layer_param_.bottom_size(); i++)
		{
			shape[2] = shape[3] = layer_param_.data_param().cropped_size(i);
			data_.push_back(new Blob<Dtype>(shape));
		}
		label_ = new Blob<int>(batch_size_, 1, 1, 1);

		label = new int[label_->count()];
		total_num_ = layer_param_.data_param().num();
		temp_num_ = 0;
		random = new unsigned int[batch_size_];

		temp_data.resize(layer_param_.bottom_size());
		temp_char.resize(layer_param_.bottom_size());
		data.resize(layer_param_.bottom_size());
		for (int i = 0; i < layer_param_.bottom_size(); i++)
		{
			temp_data[i] = new Dtype[data_[i]->count()];
			data[i] = new Dtype[data_[i]->count()];
			temp_char[i] = new char[data_[i]->count()];
			surfing_set(data_[i]->count(), (const Dtype)0.00390625, temp_data[i]);
		}

		keys.resize(batch_size_);
	}

	template <typename Dtype>
	DataLayer<Dtype>::~DataLayer()
	{
		for (int i = 0; i < data_.size(); i++)
		{
			delete data_[i];
			delete[] data[i];
			delete[] temp_char[i];
			delete[] temp_data[i];		
		}
		data_.clear();
		data.clear();
		temp_data.clear();
		temp_data.clear();

		delete[] label;
		delete[] random;
	}

	/* For different task the read data process are different */
#if defined(MNIST)
	template <typename Dtype>
	void DataLayer<Dtype>::Train_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);
		
		/*randomly get data*/
		Integer(batch_size_, random, total_num_);
	
		for (int i = 0; i < batch_size_; i++)
		{
			std::ostringstream s;
			s << std::setw(8) << std::setfill('0') << random[i];
			keys[i] = s.str();
		}
		datas = db_.GetData(keys);

		memcpy(data[0], temp_data[0], sizeof(Dtype) * data_[0]->count());

		for (int item_id = 0; item_id < batch_size_; item_id++)
		{
			datum.ParseFromString(datas[item_id]);
			int offset = data_[0]->offset(item_id);

			memcpy(temp_char[0], datum.data().c_str(), datum.data().size());

			for (int i = 0; i < datum.data().size(); i++)
			{
				data[0][offset + i] *= (int)(uint8_t)(temp_char[0][i]);
			}
			label[item_id] = datum.label();
		}
		db_.Close();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::Test_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);

		if (temp_num_ == (total_num_ / batch_size_))
		{
			temp_num_ = 0;
		}
		int temp = temp_num_ * batch_size_;
		temp_num_++;

		for (int i = 0; i < batch_size_; i++)
		{
			std::ostringstream s;
			s << std::setw(8) << std::setfill('0') << temp;
			temp++;
			keys[i] = s.str();
		}
		datas = db_.GetData(keys);

		memcpy(data[0], temp_data[0], sizeof(Dtype) * data_[0]->count());

		for (int item_id = 0; item_id < batch_size_; item_id++)
		{
			datum.ParseFromString(datas[item_id]);
			int offset = data_[0]->offset(item_id);

			memcpy(temp_char[0], datum.data().c_str(), datum.data().size());

			for (int i = 0; i < datum.data().size(); i++)
			{
				data[0][offset + i] *= (int)(uint8_t)(temp_char[0][i]);
			}
			label[item_id] = datum.label();
		}
		db_.Close();
	}

#elif defined(CIFAR10)
	template <typename Dtype>
	void DataLayer<Dtype>::Train_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);

		/*randomly get data*/
		Integer(batch_size_, random, total_num_);

		for (int i = 0; i < batch_size_; i++)
		{
			std::ostringstream s;
			s << std::setw(8) << std::setfill('0') << random[i];
			keys[i] = s.str();
		}
		datas = db_.GetData(keys);

		memcpy(data[0], temp_data[0], sizeof(Dtype) * data_[0]->count());

		for (int item_id = 0; item_id < batch_size_; item_id++)
		{
			datum.ParseFromString(datas[item_id]);
			int offset = data_[0]->offset(item_id);

			memcpy(temp_char[0], datum.data().c_str(), datum.data().size());

			for (int i = 0; i < datum.data().size(); i++)
			{
				data[0][offset + i] *= (int)(uint8_t)(temp_char[0][i]);
			}
			label[item_id] = datum.label();
		}
		db_.Close();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::Test_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);

		if (temp_num_ == (total_num_ / batch_size_))
		{
			temp_num_ = 0;
		}
		int temp = temp_num_ * batch_size_;
		temp_num_++;

		for (int i = 0; i < batch_size_; i++)
		{
			std::ostringstream s;
			s << std::setw(8) << std::setfill('0') << temp;
			temp++;
			keys[i] = s.str();
		}
		datas = db_.GetData(keys);

		memcpy(data[0], temp_data[0], sizeof(Dtype) * data_[0]->count());

		for (int item_id = 0; item_id < batch_size_; item_id++)
		{
			datum.ParseFromString(datas[item_id]);
			int offset = data_[0]->offset(item_id);

			memcpy(temp_char[0], datum.data().c_str(), datum.data().size());

			for (int i = 0; i < datum.data().size(); i++)
			{
				data[0][offset + i] *= (int)(uint8_t)(temp_char[0][i]);
			}
			label[item_id] = datum.label();
		}
		db_.Close();
	}

#elif defined(IMAGENET)
	template <typename Dtype>
	void DataLayer<Dtype>::Train_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);

		db_.Close();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::Test_Read_Batch()
	{
		db_.Open(layer_param_.data_param().source().c_str(), READ);

		db_.Close();
	}
#endif

	template <typename Dtype>
	void DataLayer<Dtype>::Set_Data(vector<Dtype*> data, int* label)
	{
		for (int i = 0; i < data.size(); i++)
		{
			data_[i]->set_cpu_data(data[i]);
		}
		label_->set_cpu_data(label);
	}



	INSTANTIATE_CLASS(DataLayer);
}
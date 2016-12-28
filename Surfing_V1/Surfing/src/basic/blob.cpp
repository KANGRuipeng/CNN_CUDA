#include "basic/blob.h"

namespace surfing
{
	template <typename Dtype>
	Blob<Dtype>::Blob(const int num, const int channels, const int height, const int width) : capacity_(0)
	{
		Reshape(num, channels, height, width);
	}

	template <typename Dtype>
	Blob<Dtype>::Blob(const vector<int>& shape) : capacity_(0)
	{
		Reshape(shape);
	}

	template <typename Dtype>
	Blob<Dtype>::~Blob()
	{
		delete data_;
		delete diff_;
	}

	template <typename Dtype>
	void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width)
	{
		vector<int> shape(4);
		shape[0] = num;
		shape[1] = channels;
		shape[2] = height;
		shape[3] = width;
		Reshape(shape);
	}

	template <typename Dtype>
	void Blob<Dtype>::Reshape(const vector<int>& shape)
	{
		count_ = 1;
		shape_.resize(shape.size());
		for (int i = 0; i < shape.size(); i++)
		{
			CHECK_GT(shape[i], 0);
			count_ *= shape[i];
			shape_[i] = shape[i];
		}
		if (count_ > capacity_)
		{
			capacity_ = count_;
			data_ = new SyncedMemory(capacity_ * sizeof(Dtype));
			diff_ = new SyncedMemory(capacity_ * sizeof(Dtype));
		}
	}

	template <typename Dtype>
	void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape)
	{
		if (reshape)
		{
			vector<int> shape(4);

			if (proto.has_num() && proto.has_channels() &&
				proto.has_height() && proto.has_width())
			{
				shape[0] = proto.num();
				shape[1] = proto.channels();
				shape[2] = proto.height();
				shape[3] = proto.width();

				Reshape(shape);
			}
			else
			{
				LOG(FATAL) << " Error in shape information !";
			}
		}
		Dtype* data_vec = mutable_cpu_data();
		if (proto.double_data_size() > 0)
		{
			CHECK_EQ(count_, proto.double_data_size());
			for (int i = 0; i < count_; i++)
			{
				data_vec[i] = proto.double_data(i);
			}
		}
		else
		{
			CHECK_EQ(count_, proto.data_size());
			for (int i = 0; i < count_; i++)
			{
				data_vec[i] = proto.data(i);
			}
		}

		if (proto.double_diff_size() > 0)
		{
			CHECK_EQ(count_, proto.double_diff_size());
			Dtype* diff_vec = mutable_cpu_diff();
			for (int i = 0; i < count_; i++)
			{
				diff_vec[i] = proto.double_diff(i);
			}
		}
		else if (proto.diff_size() > 0)
		{
			CHECK_EQ(count_, proto.diff_size());
			Dtype* diff_vec = mutable_cpu_diff();
			for (int i = 0; i < count_; i++)
			{
				diff_vec[i] = proto.diff(i);
			}
		}
	}

	template <>
	void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const
	{
		proto->set_num(shape_[0]);
		proto->set_channels(shape_[1]);
		proto->set_height(shape_[2]);
		proto->set_width(shape_[3]);

		proto->clear_data();
		proto->clear_diff();
		proto->clear_double_data();
		proto->clear_double_diff();

		const float* data_vec = cpu_data();
		for (int i = 0; i < count_/100; i++)
		{
			proto->add_data(data_vec[i * 100]);
		}
		//for (int i = 0; i < count_; i++)
		//{
		//	proto->add_data(data_vec[i]);
		//}
		if (write_diff)
		{
			const float* diff_vec = cpu_diff();
			for (int i = 0; i < count_ / 100; i++)
			{
				proto->add_diff(diff_vec[i * 100]);
			}
			//for (int i = 0; i < count_; i++)
			//{
			//	proto->add_diff(diff_vec[i]);
			//}
		}
	}
	template <>
	void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const
	{
		proto->set_num(shape_[0]);
		proto->set_channels(shape_[1]);
		proto->set_height(shape_[2]);
		proto->set_width(shape_[3]);

		proto->clear_data();
		proto->clear_diff();
		proto->clear_double_data();
		proto->clear_double_diff();

		const double* data_vec = cpu_data();
		for (int i = 0; i < count_; i++)
		{
			proto->add_double_data(data_vec[i]);
		}
		if (write_diff)
		{
			const double* diff_vec = cpu_diff();
			for (int i = 0; i < count_; i++)
			{
				proto->add_double_diff(diff_vec[i]);
			}
		}
	}

	template class Blob<int>;
	INSTANTIATE_CLASS(Blob);
}
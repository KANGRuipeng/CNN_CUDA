#ifndef BLOB_H
#define BLOB_H

#include <memory>

#include "basic/common.h"
#include "basic/syncedmemory.h"
#include "surfing.pb.h"

namespace surfing
{
	using namespace std;

	template <typename Dtype>
	class Blob
	{
	public:
		Blob() :data_(), diff_(), count_(0), capacity_(0) {}

		explicit Blob(const int num, const int channels, const int height, const int width);
		explicit Blob(const vector<int>& shape);

		~Blob();

		void Reshape(const int num, const int channels, const int height, const int width);
		void Reshape(const vector<int>& shape);

		inline const vector<int>& shape() { return shape_; }
		inline int count() { return count_; }

		inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const
		{
			CHECK_GE(n, 0);
			CHECK_LT(n, shape_[0]);
			CHECK_GE(c, 0);
			CHECK_LT(c, shape_[1]);
			CHECK_GE(h, 0);
			CHECK_LT(h, shape_[2]);
			CHECK_GE(w, 0);
			CHECK_LT(w, shape_[3]);
			return ((n * shape_[1] + c) * shape_[2] + h) * shape_[3] + w;
		}

		inline int offset(const vector<int>& indices) const
		{
			int offset = 0;
			for (int i = 0; i < shape_.size(); i++)
			{
				offset *= shape_[i];
				if (indices.size() > i)
				{
					CHECK_GT(indices[i], 0);
					CHECK_LT(indices[i], shape_[i]);
					offset += indices[i];
				}
			}
			return offset;
		}

		inline const Dtype* cpu_data() const { CHECK(data_); return (const Dtype*)data_->cpu_data(); }
		inline const Dtype* gpu_data() const { CHECK(data_); return (const Dtype*)data_->gpu_data(); }
		inline const Dtype* cpu_diff() const { CHECK(diff_); return (const Dtype*)diff_->cpu_data(); }
		inline const Dtype* gpu_diff() const { CHECK(diff_); return (const Dtype*)diff_->gpu_data(); }

		inline Dtype* mutable_cpu_data() const { CHECK(data_); return static_cast<Dtype*>(data_->mutable_cpu_data()); }
		inline Dtype* mutable_gpu_data() const { CHECK(data_); return static_cast<Dtype*>(data_->mutable_gpu_data()); }
		inline Dtype* mutable_cpu_diff() const { CHECK(diff_); return static_cast<Dtype*>(diff_->mutable_cpu_data()); }
		inline Dtype* mutable_gpu_diff() const { CHECK(diff_); return static_cast<Dtype*>(diff_->mutable_gpu_data()); }

		inline void set_cpu_data(Dtype* data) const { CHECK(data); data_->set_cpu_data(data); }
		inline void set_cpu_diff(Dtype* data) const { CHECK(data); diff_->set_cpu_data(data); }

		inline const SyncedMemory* data() const { CHECK(data_); return data_; }
		inline const SyncedMemory* diff() const { CHECK(diff_); return diff_; }

		inline Dtype data_at(const int n, const int c, const int h, const int w) const { return cpu_data()[offset(n, c, h, w)]; }
		inline Dtype diff_at(const int n, const int c, const int h, const int w) const { return cpu_diff()[offset(n, c, h, w)]; }
		inline Dtype data_at(const vector<int>& index) const { return cpu_data()[offset(index)]; }
		inline Dtype diff_at(const vector<int>& index) const { return cpu_diff()[offset(index)]; }

		void FromProto(const BlobProto& proto, bool reshape = true);
		void ToProto(BlobProto* proto, bool write_diff = false) const;

	protected:

	private:
		SyncedMemory* data_;
		SyncedMemory* diff_;

		vector<int> shape_;

		int count_;
		int capacity_;

		DISABLE_COPY_AND_ASSIGN(Blob);
	};

}

#endif
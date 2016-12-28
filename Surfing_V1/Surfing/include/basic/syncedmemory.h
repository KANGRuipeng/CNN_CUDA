#ifndef SYNCEDMEMORY_H
#define SYNCEDMEMORY_H

#include "basic/common.h"

/*In this class, the set data operation doesn't copy data, so I should be careful when use it*/
namespace surfing
{
	class SyncedMemory
	{
	public:
		enum SYNCEDHEAD { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };

		explicit SyncedMemory(size_t size) :
			cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
			own_cpu_data_(false), own_gpu_data_(false) {}

		~SyncedMemory();

		const void* cpu_data();
		const void* gpu_data();

		void set_cpu_data(void* data);

		void* mutable_cpu_data();
		void* mutable_gpu_data();

		inline SYNCEDHEAD head() { return head_; }
		inline size_t size() { return size_; }

	protected:
	private:
		void* cpu_ptr_;
		void* gpu_ptr_;

		size_t size_;
		SYNCEDHEAD head_;

		void to_cpu();
		void to_gpu();

		/*whether malloc space for data*/
		bool own_cpu_data_;
		bool own_gpu_data_;

		DISABLE_COPY_AND_ASSIGN(SyncedMemory);
	};
}
#endif
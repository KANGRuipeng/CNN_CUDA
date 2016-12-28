#include "basic/syncedmemory.h"
#include "basic/math_function.h"

namespace surfing
{
	SyncedMemory::~SyncedMemory()
	{
		if (cpu_ptr_ && own_cpu_data_){ CUDA_CHECK(cudaFreeHost(cpu_ptr_)); }
		if (gpu_ptr_ && own_gpu_data_){ CUDA_CHECK(cudaFree(gpu_ptr_)); }
	}

	void SyncedMemory::to_cpu()
	{
		switch (head_)
		{
		case UNINITIALIZED:
			CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_));
			surfing_memset(size_, 0, cpu_ptr_);
			head_ = HEAD_AT_CPU;
			own_cpu_data_ = true;
			break;
		case HEAD_AT_GPU:
			if (cpu_ptr_ == NULL)
			{
				CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_));
				own_cpu_data_ = true;
			}
			surfing_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
			head_ = SYNCED;
			break;
		case HEAD_AT_CPU:
		case SYNCED:
			break;
		}
	}

	void SyncedMemory::to_gpu()
	{
		switch (head_)
		{
		case UNINITIALIZED:
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			surfing_gpu_memset(size_, 0, gpu_ptr_);
			head_ = HEAD_AT_GPU;
			own_gpu_data_ = true;
			break;
		case HEAD_AT_CPU:
			if (gpu_ptr_ == NULL)
			{
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
				own_gpu_data_ = true;
			}
			surfing_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
			head_ = SYNCED;
			break;
		case HEAD_AT_GPU:
		case SYNCED:
			break;
		}
	}

	const void* SyncedMemory::cpu_data()
	{
		to_cpu();
		return (const void*)cpu_ptr_;
	}

	void SyncedMemory::set_cpu_data(void* data)
	{
		CHECK(data);
		if (own_cpu_data_)
		{
			memcpy(cpu_ptr_, data, size_);
		}
		else
		{
			CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_));
			memcpy(cpu_ptr_, data, size_);
		}
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
	}

	const void* SyncedMemory::gpu_data()
	{
		to_gpu();
		return (const void*)gpu_ptr_;
	}


	void* SyncedMemory::mutable_cpu_data()
	{
		to_cpu();
		head_ = HEAD_AT_CPU;
		return cpu_ptr_;
	}

	void* SyncedMemory::mutable_gpu_data()
	{
		to_gpu();
		head_ = HEAD_AT_GPU;
		return gpu_ptr_;
	}
}
#ifndef CUDNN_API_H
#define CUDNN_API_H

#include <cudnn.h>

namespace surfing
{
	namespace cudnn
	{
		template <typename Dtype> class dataType;
		template <>
		class dataType<float>
		{
		public:
			static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
			static float oneval, zeroval;
			static const void *one, *zero;
		};
		template <>
		class dataType<double>
		{
		public:
			static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
			static double oneval, zeroval;
			static const void *one, *zero;
		};


	}
}


#endif
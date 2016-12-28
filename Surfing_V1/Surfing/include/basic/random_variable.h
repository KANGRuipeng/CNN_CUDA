#ifndef RANDOM_VARIABLE_H
#define RANDOM_VARIABLE_H

namespace surfing
{
	/*here data may be device dat or host data*/
	template <typename Dtype>
	void Gaussian(size_t count, Dtype* data, Dtype mean, Dtype stddev);

	template <typename Dtype>
	void Uniform(size_t count, Dtype* data, Dtype range);

	void Integer(size_t count, unsigned int *data, unsigned int range);
}

#endif
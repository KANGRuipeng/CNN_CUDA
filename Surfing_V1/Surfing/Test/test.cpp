#include "test/rng_test.h"
#include "test/softmax_test.h"
#include "test/sigmoid_test.h"
#include "test/pooling_test.h"
#include "test/batch_normalization_test.h"
#include <iostream>

int main()
{
	Batch_Normalization_TEST();

	getchar();
	return 0;
}

#include <memory>
#include <thread>
#include <string>
#include <sys/timeb.h>
#include <iostream>

#include "glog/logging.h"

#include "net.h"
#include "solver.h"

using namespace std;
using namespace surfing;

int main()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	LOG(INFO) << " Start of program: Memory Free " << free << " Memory Total " << total;
	{
#if defined(MNIST)
		const string solver = "D:/DataBase/Mnist/parameter/solver_mnist.prototxt";
		Solver<float> surfing_solver(solver);
#elif defined(CIFAR10)
		const string solver = "D:/DataBase/Cifar10/parameter/solver_cifar10.prototxt";
		Solver<float> surfing_solver(solver);
#elif defined(IMAGENET)

#endif
		struct timeb startTime, endTime;

		ftime(&startTime);
		surfing_solver.Train();
		ftime(&endTime);
		LOG(INFO) << " Run time is " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << " mill second ";

		ftime(&startTime);
		surfing_solver.Test();
		ftime(&endTime);
		LOG(INFO) << " Run time is " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << " mill second ";
	}

	cudaMemGetInfo(&free, &total);
	LOG(INFO) << " End of program: Memory Free " << free << " Memory Total " << total;

	getchar();
	return 0;
}
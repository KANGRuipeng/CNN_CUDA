#ifndef SOLVER_H
#define SOLVER_H

#include "basic/common.h"
#include "basic/io_parameter.h"
#include "net.h"
#include "surfing.pb.h"

namespace surfing
{
	template <typename Dtype>
	class Solver
	{
	public:
		explicit Solver(const string& param_file);
		~Solver();

		inline const SolverParameter& solver_param() { return solver_param_; }

		void Net_Setup();
		void Train();
		void Test();

	protected:

	private:
		SolverParameter solver_param_;
		Net<Dtype>* net_;

		int max_iter_;
		int test_epoch_;
		Dtype global_lr_;
		Dtype momentum_;

		DataLayer<Dtype>* train_data_io_;
		DataLayer<Dtype>* test_data_io_;

		DISABLE_COPY_AND_ASSIGN(Solver);
	};

}
#endif
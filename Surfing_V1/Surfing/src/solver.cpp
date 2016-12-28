#include "solver.h"
#include <thread>
#include <iomanip>
#include <fstream>
#include <sys/timeb.h>

namespace surfing
{
	template <typename Dtype>
	Solver<Dtype>::Solver(const string& param_file)
	{
		ReadProtoFromTextFile(param_file.c_str(), &solver_param_);
		LOG(INFO) << " Initializing solver from parameters: " << std::endl << solver_param_.DebugString();
		
		max_iter_ = solver_param_.max_iter();
		global_lr_ = solver_param_.global_learning_rate();
		momentum_ = solver_param_.momentum();
		test_epoch_ = solver_param_.test_epoch();

		/*This function call the net constructor*/
		Net_Setup();
	}

	template <typename Dtype>
	Solver<Dtype>::~Solver()
	{
		delete net_;
		delete train_data_io_;
		delete test_data_io_;
	}


	template <typename Dtype>
	void Solver<Dtype>::Net_Setup()
	{
		if (solver_param_.is_pre_train() == SolverParameter::NEW)
		{
			net_ = new Net<Dtype>(solver_param_.net(), false);
		}
		else
		{
			net_ = new Net<Dtype>(solver_param_.net_binary(), true);
		}
		/*new two layer used for data io seperately*/
		train_data_io_ = new DataLayer<Dtype>(net_->train_data_param());
		test_data_io_ = new DataLayer<Dtype>(net_->test_data_param());
	}

	template <typename Dtype>
	void Solver<Dtype>::Train()
	{
		int epoch = 0;

		std::thread train_data(std::bind(&DataLayer<Dtype>::Train_Read_Batch, train_data_io_));
		train_data.join();
		net_->Train_Set_Data(train_data_io_->Temp_Data(), train_data_io_->Temp_Label());

		int j = 0;
		for (int i = 0; i < max_iter_; i++)
		{
			std::thread train_data(std::bind(&DataLayer<Dtype>::Train_Read_Batch, train_data_io_));
			net_->Train(global_lr_, momentum_);
			train_data.join();

			net_->Train_Set_Data(train_data_io_->Temp_Data(), train_data_io_->Temp_Label());

			epoch++;
			if (epoch == solver_param_.display_epoch())
			{
				epoch = 0;
				Test();

				std::ostringstream s;
				s << std::setw(8) << std::setfill('0') << i;
#if defined(MNIST)
				string filename = "D:/DataBase/Mnist/parameter/" + s.str() + ".prototxt";
				net_->ToProto(filename.c_str(), true);
#elif defined(CIFAR10)
				string filename = "D:/DataBase/Cifar10/parameter/" + s.str() + ".prototxt";
				net_->ToProto(filename.c_str(), true);
#elif defined(IMAGENET)

#endif
			}
			//LOG(INFO) << " epoch " << i;
		}


	}

	template <typename Dtype>
	void Solver<Dtype>::Test()
	{
		std::thread test_data(std::bind(&DataLayer<Dtype>::Test_Read_Batch, test_data_io_));
		test_data.join();
		net_->Test_Set_Data(test_data_io_->Temp_Data(), test_data_io_->Temp_Label());
		net_->Set_Loss();
		for (int i = 0; i < test_epoch_ - 1; i++)
		{
			std::thread test_data(std::bind(&DataLayer<Dtype>::Test_Read_Batch, test_data_io_));
			net_->Test();
			test_data.join();
			net_->Test_Set_Data(test_data_io_->Temp_Data(), test_data_io_->Temp_Label());
		}
		net_->Test();
		LOG(INFO) << " Loss " << net_->Get_Loss();
 		LOG(INFO) << " Accurate " << net_->Get_Accurate_Count();
		net_->Set_Accurate_Count();
	}

	INSTANTIATE_CLASS(Solver);
}
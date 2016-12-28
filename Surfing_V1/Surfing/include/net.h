#ifndef NET_H
#define NET_H

#include "basic/io_parameter.h"
#include "basic/common.h"
#include "surfing.pb.h"

#include "layer/layer.h"
#include "layer/data_layer.h"
#include "layer/result_layer.h"

namespace surfing
{
	using namespace std;

	template <typename Dtype>
	class Net
	{
	public:
		explicit Net(const string& param_file, bool pre_train);
		~Net();

		inline const NetParameter& net_param() { return net_param_; }

		inline const LayerParameter& train_data_param() const  { return train_data_layer_->layer_param(); }
		inline const LayerParameter& test_data_param() const  { return test_data_layer_->layer_param(); }

		void Set_Accurate_Count() { test_result_layer_->Set_Accurate_Count(); }
		int Get_Accurate_Count() { return  test_result_layer_->Get_Accurate_Count(); }
		void Set_Loss() { test_result_layer_->Set_Loss(); }
		Dtype Get_Loss() { return  test_result_layer_->Get_Loss(); }

		void Train_Set_Data(vector<Dtype*> data, int* label) { train_data_layer_->Set_Data(data, label); }
		void Test_Set_Data(vector<Dtype*> data, int* label) { test_data_layer_->Set_Data(data, label); }

		void Train(Dtype global_lr, Dtype momentum)
		{
			Train_Forward();
			Train_Backward();
			Apply_Gradient(global_lr, momentum);
		}
		void Test() { Test_Forward(); }

		void ToProto(const char* filename, bool write_diff = false);
		void FromProto(const char* filename, bool reshape = false);

	protected:
	private:		
		void Init();
		void Init_Desc();
		Layer<Dtype>* Init_Layers(const LayerParameter& param);

		void Train_Forward();
		void Train_Backward();
		void Apply_Gradient(Dtype global_lr, Dtype momentum);
		void Test_Forward();

		NetParameter net_param_;

		/*Only save common layer, input and output for train and test are set separately*/
		vector<Layer<Dtype>*> layers_;
		map<string, int> layer_names_id_;
		map<string, int> layer_names_priority_;
		/* This is used to set up the priority*/
		map<string, int> blob_names_priority_;
		vector<vector<string>> layer_priority_;

		/* Sepcial layers */
		DataLayer<Dtype>* train_data_layer_;
		DataLayer<Dtype>* test_data_layer_;
		ResultLayer<Dtype>* train_result_layer_;
		ResultLayer<Dtype>* test_result_layer_;

		/* This is used to store real blobs used*/
		vector<Blob<Dtype>*> blobs_;
		map<string, int> blob_names_id_;

		/* Used to build different layers input */
		map<string, int> train_input_names_id_;
		map<string, int> test_input_names_id_;

		vector<vector<Blob<Dtype>*>> train_bottom_blobs_;
		vector<Blob<Dtype>*> train_top_blobs_;

		vector<vector<Blob<Dtype>*>> test_bottom_blobs_;
		vector<Blob<Dtype>*> test_top_blobs_;

		/* Those parameters are used for update */
		cublasHandle_t cublas_;
		Dtype filter_lr_;
		Dtype bias_lr_;

		DISABLE_COPY_AND_ASSIGN(Net);
	};

}
#endif
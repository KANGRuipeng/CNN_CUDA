#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include "layer/layer.h"
#include "basic/dblmdb.h"

namespace surfing
{
	template <typename Dtype>
	class DataLayer : public Layer<Dtype>
	{
	public:
		DataLayer(const LayerParameter& param);
		~DataLayer();

		void Train_Read_Batch();
		void Test_Read_Batch();
		void Set_Data(vector<Dtype*> data, int* label);

		inline int Batch_Size() { return batch_size_; }
		
		inline vector<Dtype*>& Temp_Data() { return data; }
		inline int*& Temp_Label() { return label; }

		inline vector<Blob<Dtype>*>& Data() { return data_; }
		inline Blob<int>*& Label() { return label_; }
	
	private:		
		vector<Blob<Dtype>*> data_;
		Blob<int>* label_;

		/* temp data used to transform data */
		vector<Dtype*> data;
		vector<Dtype*> temp_data;
		vector<char*> temp_char;
		int* label;

		/*this variable used to randomly get data*/
		unsigned int* random;

		/* Database IO parameter */
		Datum datum;
		LMDB db_;
		vector<string> keys, datas;
		
		int batch_size_;
		int total_num_;
		/* This is flag variable used for test. */
		int temp_num_;
	};
}
#endif
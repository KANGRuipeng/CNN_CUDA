#include "data/image_mean.h"
#include "basic/blob.h"
#include "basic/io_parameter.h"

#include "data/image_crop.h"
#include "glog/logging.h"
#include "surfing.pb.h"
#include "basic/dblmdb.h"

#include <fstream>
#include <iomanip>

namespace surfing
{
	using namespace cv;
	void imagenet_mean(string data,string destination)
	{
		Blob<float> mean_blob(1, 3, PICTURE_SIZE, PICTURE_SIZE);

		unsigned long long int* sum = new unsigned long long int[3 * PICTURE_SIZE * PICTURE_SIZE];
		float *mean = new float[3 * PICTURE_SIZE * PICTURE_SIZE];
		memset((void*)sum, 0, 3 * PICTURE_SIZE * PICTURE_SIZE * sizeof(unsigned long long int));

		surfing::LMDB db_;
		surfing::Datum datum;

		int batch_size = 50;

		for (int j = 0; j < 25623; j++)
		{
			vector<string> keys, datas;				
			for (int i = 0; i < batch_size; i++)
			{
				std::ostringstream s;
				s << std::setw(8) << std::setfill('0') << j * batch_size + i;			
				keys.push_back(s.str());	
				//LOG(INFO) << s.str();
			}

			db_.Open(data.c_str(), surfing::READ);
			datas = db_.GetData(keys);
			db_.Close();

			for (int i = 0; i < batch_size; i++)
			{
				datum.ParseFromString(datas[i]);
				for (int k = 0; k < datum.data().size(); k++)
				{
					sum[k] += (int)(uint8_t)datum.data()[k];
				}
			}

		}
		std::ofstream mean_file(destination, std::fstream::out);

		for (int k = 0; k < 3 * PICTURE_SIZE * PICTURE_SIZE; k++)
		{
			mean[k] = (float)sum[k] / (256.0 * 50 * 25623);
			mean_file << mean[k] << " " << sum[k] <<std::endl;
		}

		mean_blob.set_cpu_data(mean);
		BlobProto* param = new BlobProto;
		mean_blob.ToProto(param, false);
		WriteProtoToTextFile(*param, "D:/DataBase/ImageNet/ImageNet/mean_file.prototxt");
		WriteProtoToBinaryFile(*param, "D:/DataBase/ImageNet/ImageNet/mean_file.binary");
	}
}

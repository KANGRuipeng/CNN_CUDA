#include "data/display.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "glog/logging.h"
#include "surfing.pb.h"
#include "basic/dblmdb.h"

namespace surfing
{
	/*This function can be used to display well resized data*/
	void display(string destination,int num)
	{
		using namespace cv;

		surfing::LMDB db_;
		surfing::Datum datum;

		db_.Open(destination.c_str(), surfing::READ);

		std::ostringstream s;
		s << std::setw(8) << std::setfill('0') << num;
		std::string key_str = s.str();
		
		vector<string> keys, datas;
		keys.push_back(key_str);
		datas = db_.GetData(keys);
		datum.ParseFromString(datas[0]);

		LOG(INFO) << datum.label();
		
		Mat img(datum.height(), datum.width(), CV_8UC3);
		
		for (int j = 0; j < datum.height(); j++)
		{
			for (int k = 0; k < datum.width(); k++)
			{
				if (datum.channels() == 3)
				{					
					img.at<unsigned char>(j, k * 3) = datum.data()[j * datum.height() + k];
					img.at<unsigned char>(j, k * 3 + 1) = datum.data()[datum.height() * datum.width() + j * datum.height() + k];
					img.at<unsigned char>(j, k * 3 + 2) = datum.data()[2 * datum.height() * datum.width() + j * datum.height() + k];										
				}
				else
				{
					img.at<unsigned char>(j, k*3) = datum.data()[k * datum.height() + j];
					img.at<unsigned char>(j, k * 3 + 1) = datum.data()[k * datum.height() + j];
					img.at<unsigned char>(j, k * 3 + 2) = datum.data()[k * datum.height() + j];
				}
			}
		}

		imshow("Display", img);
		waitKey(0);

		db_.Close();
	}

	/* This function can be used to display mat format image */
	void display_mat(string destination, int num)
	{
		using namespace cv;

		surfing::LMDB db_;
		surfing::Datum datum;

		db_.Open(destination.c_str(), surfing::READ);

		std::ostringstream s;
		s << std::setw(8) << std::setfill('0') << num;
		std::string key_str = s.str();

		vector<string> keys, datas;
		keys.push_back(key_str);
		datas = db_.GetData(keys);

		datum.ParseFromString(datas[0]);
		LOG(INFO) << datum.label();

		Mat img(datum.height(), datum.width(), CV_8UC3);
		memcpy(img.data, datum.data().c_str(), datum.data().size());

		imshow("Display", img);
		waitKey(0);

		db_.Close();
	}
}
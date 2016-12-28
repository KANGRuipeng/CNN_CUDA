#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iomanip>

#include "glog/logging.h"
#include "surfing.pb.h"
#include "basic/dblmdb.h"
#include "data/xml_parse.h"
#include "data/image_crop.h"
#include "data/image_mean.h"

using std::string;
using namespace cv;
using namespace surfing;

void convert_train(string train_name_file, string prefix_train, string train_destination, string label_number);
void convert_val(string val_name_file, string prefix_val_annotation,
	string prefix_val_data, string val_destination, string label_number);


int main()
{
	string train_name_file = "D:/DataBase/ImageNet/ILSVRC2015_CLS-LOC/ILSVRC2015/ImageSets/CLS-LOC/train_cls.txt";
	string val_name_file = "D:/DataBase/ImageNet/ILSVRC2015_CLS-LOC/ILSVRC2015/ImageSets/CLS-LOC/val.txt";

	string prefix_train = "D:/DataBase/ImageNet/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train/";
	string prefix_val_annotation = "D:/DataBase/ImageNet/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val/";
	string prefix_val_data = "D:/DataBase/ImageNet/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val/";

	string train_destination = "D:/DataBase/ImageNet/ImageNet/train_lmdb";
	string val_destination = "D:/DataBase/ImageNet/ImageNet/val_lmdb";

	string label_number = "D:/DataBase/ImageNet/ImageNet/label_number.txt";

	convert_train(train_name_file, prefix_train, train_destination, label_number);
	convert_val(val_name_file, prefix_val_annotation, prefix_val_data, val_destination, label_number);

	imagenet_mean("D:/DataBase/ImageNet/ImageNet/train_lmdb", "D:/DataBase/ImageNet/ImageNet/mean_file.txt");

	return 0;
}

void convert_train(string train_name_file, string prefix_train, string train_destination, string label_number)
{
	string suffix = ".JPEG";

	std::ifstream names(train_name_file, std::ios::in);
	std::ofstream labels(label_number, std::fstream::out);

	if (!names.is_open())
	{
		LOG(FATAL) << " file open error";
	}
	if (!labels.is_open())
	{
		LOG(FATAL) << " file open error";
	}
	/*buffer use to save strings to get the file name*/
	string name;
	char buffer[128], buffer0[128], buffer1[128];
	/* use opencv to read data and transform to proper shape*/
	Mat cropped;

	/*Used to match label*/
	std::map<string, int> label_to_number;
	std::map<string, int>::iterator it;
	/* used to count label num */
	int size; 

	surfing::Datum datum;
	string value;
	int count = 0;
	datum.set_channels(3);
	datum.set_height(PICTURE_SIZE);
	datum.set_width(PICTURE_SIZE);

	int pixel = 3 * PICTURE_SIZE * PICTURE_SIZE;

	/* database related */
	surfing::LMDB db;
	db.Open(train_destination, surfing::NEW);
	surfing::LMDBTransaction* txn(db.NewTransaction());

	/* i = 1281167 */
	for (int item_id = 0; item_id < 1281167; item_id++)
	{
		names.getline(buffer, 128, '/');
		names.getline(buffer0, 128, ' ');
		names.getline(buffer1, 128, '\n');
		name = prefix_train + buffer + "/" + buffer0 + suffix;

		it = label_to_number.find(buffer);
		if (it == label_to_number.end())
		{
			size = label_to_number.size() + 1;
			label_to_number.insert(std::pair<string, int>(buffer, size));
			labels << buffer << " " << size << "\n";
		}

		cropped = Cropped_Image(name);

		datum.set_data(cropped.data, pixel);
		datum.set_label(label_to_number.find(buffer)->second);

		std::ostringstream s;
		s << std::setw(8) << std::setfill('0') << item_id;

		std::string key_str = s.str();
		datum.SerializeToString(&value);

		txn->Put(key_str, value);
		if (++count % 1000 == 0)
		{
			txn->Commit();
		}
	}
	if (count % 1000 != 0)
	{
		txn->Commit();
	}

	LOG(INFO) << "Processed " << count << " files";
	db.Close();
}

void convert_val(string val_name_file, string prefix_val_annotation, 
	string prefix_val_data, string val_destination, string label_number)
{
	string suffix = ".xml";
	string suffix_image = ".JPEG";

	std::ifstream names(val_name_file, std::ios::in);
	string name;
	
	char buffer[128], buffer0[128];
	for (int i = 0; i < 128; i++)
	{
		buffer0[i] = 0;
	}

	std::ifstream label_file(label_number, std::ios::in);
	std::map<string, int> labels;
	int number = 0;
	for (int item_id = 0; item_id < 1000; item_id++)
	{
		number = 0;
		label_file.getline(buffer, 128, ' ');
		label_file.getline(buffer0, 128, '\n');
		for (int i = 0; buffer0[i] != 0; i++)
		{
			number = number * 10 + (buffer0[i] - '0');
		}
		labels.insert(std::pair<string, int>(buffer, number));
	}

	surfing::LMDB db;
	db.Open(val_destination, surfing::NEW);
	surfing::LMDBTransaction* txn(db.NewTransaction());

	string picture_name, label;

	surfing::Datum datum;
	std::string value;
	int count = 0;
	datum.set_channels(3);
	datum.set_height(PICTURE_SIZE);
	datum.set_width(PICTURE_SIZE);

	int pixel = 3 * PICTURE_SIZE * PICTURE_SIZE;

	Mat cropped;

	for (int item_id = 0; item_id < 50000; item_id++)
	{
		int object_number = 0;
		names.getline(buffer, 128, ' ');
		names.getline(buffer0, 128, '\n');

		name = prefix_val_annotation + buffer + suffix;
		
		Val_Label(name, picture_name, label);

		string address = prefix_val_data + picture_name + suffix_image;

		cropped = Cropped_Image(address);

		datum.set_data(cropped.data, pixel);
		datum.set_label(labels.find(label)->second);

		std::ostringstream s;
		s << std::setw(8) << std::setfill('0') << item_id;

		std::string key_str = s.str();
		datum.SerializeToString(&value);

		txn->Put(key_str, value);
		if (++count % 1000 == 0)
		{
			txn->Commit();
		}
	}
	if (count % 1000 != 0)
	{
		txn->Commit();
	}
	LOG(INFO) << "number of data handled" << count;
	db.Close();
}

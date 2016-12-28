#include <fstream>
#include <memory>
#include <iomanip>
#include <vector>
#include <string>

#include "glog/logging.h"
#include "surfing.pb.h"
#include "basic/dblmdb.h"

using namespace std;
void convert_dataset(vector<string> source_filename, const char* db_path);

int main()
{
	vector<string> train_file;
	train_file.push_back("D:/DataBase/Cifar10/data_batch_1.bin");
	train_file.push_back("D:/DataBase/Cifar10/data_batch_2.bin");
	train_file.push_back("D:/DataBase/Cifar10/data_batch_3.bin");
	train_file.push_back("D:/DataBase/Cifar10/data_batch_4.bin");
	train_file.push_back("D:/DataBase/Cifar10/data_batch_5.bin");
	string train_path = "D:/DataBase/Cifar10/cifar10_train_lmdb";

	vector<string> test_file;
	test_file.push_back("D:/DataBase/Cifar10/test_batch.bin");
	string test_path = "D:/DataBase/Cifar10/cifar10_test_lmdb";

	convert_dataset(train_file, train_path.c_str());
	convert_dataset(test_file, test_path.c_str());

	return 0;
}

void convert_dataset(vector<string> source_filename, const char* db_path)
{
	surfing::LMDB db;
	db.Open(db_path, surfing::NEW);
	surfing::LMDBTransaction* txn(db.NewTransaction());

	surfing::Datum datum;
	datum.set_channels(3);
	datum.set_height(32);
	datum.set_width(32);

	char label;
	char pixel[3072];
	std::string value;
	int count = 0;

	for (int file_id = 0; file_id < source_filename.size(); file_id++)
	{
		std::ifstream source_file(source_filename[file_id].c_str(), std::ios::in | std::ios::binary);
		
		
		for (int item_id = 0; item_id < 10000; item_id++)
		{
			source_file.read(&label, 1);
			source_file.read(pixel,3072);

			datum.set_data(pixel, 3072);
			datum.set_label(label);

			std::ostringstream s;
			s << std::setw(8) << std::setfill('0') << file_id * 10000 + item_id;

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
	}
	LOG(INFO) << "Processed " << count << " files";
	db.Close();
}
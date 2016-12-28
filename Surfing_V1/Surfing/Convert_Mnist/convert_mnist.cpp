#include <fstream>
#include <memory>
#include <iomanip>

#include "glog/logging.h"
#include "surfing.pb.h"
#include "basic/dblmdb.h"

using namespace surfing;

uint32_t swap_endian(uint32_t val);
void convert_dataset(const char* image_filename, const char* label_filename, const char* db_path);

int main(int argc, char** argv)
{
	//google::InitGoogleLogging(argv[0]);
	char a[] = "D:/DataBase/Mnist/train-images-idx3-ubyte";
	char b[] = "D:/DataBase/Mnist/train-labels-idx1-ubyte";
	char c[] = "D:/DataBase/Mnist/mnist_train_lmdb";
	char d[] = "D:/DataBase/Mnist/t10k-images-idx3-ubyte";
	char e[] = "D:/DataBase/Mnist/t10k-labels-idx1-ubyte";
	char f[] = "D:/DataBase/Mnist/mnist_test_lmdb";
	convert_dataset(a, b, c);
	convert_dataset(d, e, f);
	return 0;
}

void convert_dataset(const char* image_filename, const char* label_filename, const char* db_path)
{
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << " Unable to open file " << image_filename;
	CHECK(label_file) << " Unable to open file " << label_filename;

	uint32_t magic;
	uint32_t num_images;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic";
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic";

	image_file.read(reinterpret_cast<char*>(&num_images), 4);
	num_images = swap_endian(num_images);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_images, num_labels);

	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

	surfing::LMDB db;
	db.Open(db_path, surfing::NEW);
	surfing::LMDBTransaction* txn(db.NewTransaction());

	char label;
	uint32_t pixel = rows*cols;
	LOG(INFO) << pixel;
	char* pixels = new char[pixel];
	int count = 0;
	std::string value;

	surfing::Datum datum;

	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);

	LOG(INFO) << "A total of " << num_images << " iamges.";
	LOG(INFO) << "Rows " << rows << " Cols " << cols;

	for (int item_id = 0; item_id < num_images; item_id++)
	{
		image_file.read(pixels, pixel);
		label_file.read(&label, 1);
		datum.set_data(pixels, pixel);
		datum.set_label(label);

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
	delete[] pixels;
	db.Close();
}

uint32_t swap_endian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
	return (val << 16) | (val >> 16);
}


#include "basic/io_parameter.h"

#include <google/protobuf/io/coded_stream.h>

#include <fstream>

namespace surfing
{
	bool ReadProtoFromTextFile(const char* filename, Message* proto)
	{
		int fd = open(filename, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		close(fd);
		return success;
	}

	void WriteProtoToTextFile(const Message& proto, const char* filename)
	{
		int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		FileOutputStream* output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(proto, output));
		delete output;
		close(fd);
	}


	bool ReadProtoFromBinaryFile(const char* filename, Message* proto)
	{
		int fd = open(filename, O_RDONLY | O_BINARY);
		CHECK_NE(fd, -1) << "File not find: " << filename;

		ZeroCopyInputStream* raw_input = new FileInputStream(fd);
		CodedInputStream* coded_input = new CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

		bool success = proto->ParseFromCodedStream(coded_input);

		delete coded_input;
		delete raw_input;
		close(fd);
		return success;
	}

	void WriteProtoToBinaryFile(const Message& proto, const char* filename)
	{
		using namespace std;
		fstream output(filename, ios::out | ios::trunc | ios::binary);
		CHECK(proto.SerializeToOstream(&output));
	}

}

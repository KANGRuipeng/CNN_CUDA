#ifndef IO_PARAMETER_H
#define IO_PARAMETER_H

#include <io.h>
#include <fcntl.h>

#include "google/protobuf/message.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "glog/logging.h"

namespace surfing
{
	using google::protobuf::Message;
	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;

	bool ReadProtoFromTextFile(const char* filename, Message* proto);
	void WriteProtoToTextFile(const Message& proto, const char* filename);

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto);
	void WriteProtoToBinaryFile(const Message& proto, const char* filename);
}


#endif
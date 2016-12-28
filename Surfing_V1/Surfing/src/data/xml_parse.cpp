#include "data/xml_parse.h"
#include "glog/logging.h"

#include "tinyxml2.h"

namespace surfing
{
	void Val_Label(string name, string& picture_name, string& label)
	{
		tinyxml2::XMLDocument doc;
		if (!doc.LoadFile(name.c_str()))
		{
			DLOG(INFO) << " load file successfully !";
		}
		else
		{
			LOG(FATAL) << " load file failed !";
		}

		tinyxml2::XMLElement* element = doc.FirstChildElement("annotation");
		tinyxml2::XMLElement* child;


		for (child = element->FirstChildElement(); child != NULL; child = child->NextSiblingElement())
		{
			int object_number = 0;

			string type = child->Name();
			if (type == "folder")
			{
			}
			else if (type == "filename")
			{
				picture_name = child->GetText();
			}
			else if (type == "source")
			{
			}
			else if (type == "size")
			{
			}
			else if (type == "segmented")
			{
			}
			else if (type == "object")
			{
				if (!object_number)
				{
					object_number++;
					label = child->FirstChildElement()->GetText();
				}
				else
				{
					string label2 = child->FirstChildElement()->GetText();
					if (*label2.c_str() != *label.c_str())
					{
						LOG(FATAL) << " Multi labels !";
					}
				}
			}
		}
	}

}

#ifndef IMAGE_CROP_H
#define IMAGE_CROP_H

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PICTURE_SIZE 256 
using namespace cv;

namespace surfing
{
	Mat Cropped_Image(std::string name);
}

#endif
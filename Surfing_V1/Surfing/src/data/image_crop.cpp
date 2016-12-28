#include "data/image_crop.h"

namespace surfing
{
	Mat Cropped_Image(std::string name)
	{
		Mat srcImage, resizeImage, cropped, cropedImage;

		srcImage = imread(name.c_str());

		int w = srcImage.rows;
		int h = srcImage.cols;

		if (h > w)
		{
			h = (int)(h * PICTURE_SIZE / w);
			resize(srcImage, resizeImage, Size(h, PICTURE_SIZE), 0, 0, INTER_LANCZOS4);
			cropedImage = resizeImage(Rect((int)((h - PICTURE_SIZE) / 2), 0, PICTURE_SIZE, PICTURE_SIZE));
		}
		else
		{
			w = (int)(w * PICTURE_SIZE / h);
			resize(srcImage, resizeImage, Size(PICTURE_SIZE, w), 0, 0, INTER_LANCZOS4);
			cropedImage = resizeImage(Rect(0, (int)((w - PICTURE_SIZE) / 2), PICTURE_SIZE, PICTURE_SIZE));
		}
		cropedImage.copyTo(cropped);
		return cropped;
	}

}
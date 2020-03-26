#include "SEETAPassiveFaceAntiSpoofing.h"
#include <string>
#include "seeta/Struct.h"
#include "seeta/ImageProcess.h"
#include "seeta/common_alignment.h"
#include <SeetaStruct.h>
#include <HolidayForward.h>

// SEETAPassiveFaceAntiSpoofing::SEETAPassiveFaceAntiSpoofing(const std::string ModelPath)
// {
// 	std::string modelName2timesFace("_acc97_res14YuvAlignment_68000_0622");
// 	output_blob_name2timesFace = "prob";
// 	char* buffer2timesFace = nullptr;
// 	int64_t length2timesFace;
// 	SeetaReadAllContentFromFile((ModelPath + modelName2timesFace + ".bin").c_str(), &buffer2timesFace, &length2timesFace);
// 	SeetaCNN_Model *model2timesFace = nullptr;
// 	SeetaReadModelFromBuffer(buffer2timesFace, length2timesFace, &model2timesFace);
// 	net2timesFace = nullptr;
// 	SeetaCreateNet(model2timesFace, 1, HOLIDAY_CNN_CPU_DEVICE, &net2timesFace);
// 	SeetaReleaseModel(model2timesFace);
// 	SeetaFreeBuffer(buffer2timesFace);
// 	SeetaNetKeepBlob(net2timesFace, output_blob_name2timesFace);
// }

SEETAPassiveFaceAntiSpoofing::SEETAPassiveFaceAntiSpoofing(
    const SEETAPassiveFASAndShakeHead::ModelSetting& model_setting)
    : SEETAPassiveFaceAntiSpoofing(model_setting.FAS)
{
}

SEETAPassiveFaceAntiSpoofing::SEETAPassiveFaceAntiSpoofing(const std::string& model_path)
{
    // TODO: simple codes
    output_blob_name2timesFace = "prob";
    char* buffer2timesFace = nullptr;
    int64_t length2timesFace;
    SeetaReadAllContentFromFile(model_path.c_str(), &buffer2timesFace, &length2timesFace);
    load(SEETAPassiveFASAndShakeHead::Buffer(buffer2timesFace, length2timesFace));
    SeetaFreeBuffer(buffer2timesFace);
}

SEETAPassiveFaceAntiSpoofing::SEETAPassiveFaceAntiSpoofing(const SEETAPassiveFASAndShakeHead::Buffer& model_buffer)
{
    output_blob_name2timesFace = "prob";
    load(model_buffer);
}

void SEETAPassiveFaceAntiSpoofing::load(const SEETAPassiveFASAndShakeHead::Buffer& buffer)
{

    SeetaCNN_Model *model2timesFace = nullptr;
    SeetaReadModelFromBuffer(reinterpret_cast<const char *>(buffer.data), buffer.size, &model2timesFace);
    net2timesFace = nullptr;
    SeetaCreateNet(model2timesFace, 1, HOLIDAY_CNN_CPU_DEVICE, &net2timesFace);
    SeetaReleaseModel(model2timesFace);
    SeetaNetKeepBlob(net2timesFace, output_blob_name2timesFace);
}

SEETAPassiveFaceAntiSpoofing::~SEETAPassiveFaceAntiSpoofing()
{
	if (net2timesFace) SeetaReleaseNet(net2timesFace);
}

static double matToTheSeetaNet(const seeta::Image &inputImage, const int inputChannel,
	const int inputHeight, const int inputWidth,
	SeetaCNN_Net* net, const char* output_blob_name)
{
	SeetaCNN_InputOutputData tmp_input;
	tmp_input.number = 1;
	tmp_input.channel = inputChannel;
	tmp_input.height = inputHeight;
	tmp_input.width = inputWidth;
	tmp_input.data_point_char = const_cast<unsigned char*>(inputImage.data());
	tmp_input.buffer_type = SEETACNN_BGR_IMGE_CHAR;

	SeetaRunNetChar(net, 1, &tmp_input);

	SeetaCNN_InputOutputData tmp_output;
	SeetaGetFeatureMap(net, output_blob_name, &tmp_output);
	return tmp_output.data_point_float[1];
}

static seeta::Image copyMakeBorder(const seeta::Image &img, int top, int bottom, int left, int right)
{
	if (top == 0 && bottom == 0 && left == 0 && right == 0) return img;
	seeta::Image result(img.width() + left + right, img.height() + top + bottom, img.channels());
	seeta::fill(result, seeta::Point(left, top), img);
	if (img.width() == 0 || img.height() == 0) return result;
	// left
	for (int y = top; y < top + img.height(); ++y)
	{
		for (int x = 0; x < left; ++x)
		{
			std::memcpy(&result.data(y, x, 0), &result.data(y, left, 0), result.channels());
		}
	}
	// right
	for (int y = top; y < top + img.height(); ++y)
	{
		for (int x = left + img.width(); x < result.width(); ++x)
		{
			std::memcpy(&result.data(y, x, 0), &result.data(y, left + img.width() - 1, 0), result.channels());
		}
	}
	// top
	for (int y = 0; y < top; ++y)
	{
		std::memcpy(&result.data(y, 0, 0), &result.data(top, 0, 0), result.width() * result.channels());
	}
	// bottom
	for (int y = top + img.height(); y < result.height(); ++y)
	{
		std::memcpy(&result.data(y, 0, 0), &result.data(top + img.height() - 1, 0, 0), result.width() * result.channels());
	}
	return result;

}

static seeta::Image ImageFillBorder(const seeta::Image &img, const seeta::Rect &rect){
	// seeta::Image imgFill(rect.height, rect.width, 0);// 目标图像  
	// imgFill.for_each([](seeta::Image::Datum &pixel){pixel = 0; });
	// 获取可填充图像  
	int crop_x1 = std::max(0, rect.x);
	int crop_y1 = std::max(0, rect.y);
	int crop_x2 = std::min(img.width() - 1, rect.x + rect.width - 1);	// 图像范围 0到cols-1, 0到rows-1  
	int crop_y2 = std::min(img.height() - 1, rect.y + rect.height - 1);
	seeta::Image roi_img = seeta::crop(img, seeta::Region(crop_y1, crop_y2, crop_x1, crop_x2));	// 左包含，右不包含  
	//return roi_img; 
	// 如果需要填边  
	int left_x = (-rect.x);
	int top_y = (-rect.y);
	int right_x = rect.x + rect.width - img.width();
	int down_y = rect.y + rect.height - img.height();
	left_x = (left_x > 0 ? left_x : 0);
	right_x = (right_x > 0 ? right_x : 0);
	top_y = (top_y > 0 ? top_y : 0);
	down_y = (down_y > 0 ? down_y : 0);
	auto imgFill = copyMakeBorder(roi_img, top_y, down_y, left_x, right_x);
	// 自带填充边界函数，top_y, down_y, left_x, right_x为非负正数  
	// 而且imgFill.cols = roi_img.cols + left_x + right_x, imgFill.rows = roi_img.rows + top_y + down_y  
	return imgFill;
}
static bool CropFace(const seeta::Image& srcImg, const VIPLPoint *llpoint, seeta::Image& dstImg, VIPLPoint *finalPoint = nullptr)
{
	const int cropFaceSize = 512;

	if (dstImg.width() != cropFaceSize || dstImg.height() != cropFaceSize || dstImg.channels() != srcImg.channels())
	{
		dstImg = seeta::Image(cropFaceSize, cropFaceSize, srcImg.channels());
	}

	float mean_shape[10] = {
		89.3095f + cropFaceSize / 4,
		72.9025f + cropFaceSize / 4,
		169.3095f + cropFaceSize / 4,
		72.9025f + cropFaceSize / 4,
		127.8949f + cropFaceSize / 4,
		127.0441f + cropFaceSize / 4,
		96.8796f + cropFaceSize / 4,
		184.8907f + cropFaceSize / 4,
		159.1065f + cropFaceSize / 4,
		184.7601f + cropFaceSize / 4,
	};
	float points[10];
	for (int i = 0; i < 5; ++i)
	{
		points[2 * i] = float(llpoint[i].x);
		points[2 * i + 1] = float(llpoint[i].y);
	}
	float tempPoint[10];

	face_crop_core(srcImg.data(), srcImg.width(), srcImg.height(), srcImg.channels(), dstImg.data(), cropFaceSize, cropFaceSize, points, 5, mean_shape, cropFaceSize, cropFaceSize, 0, 0, 0, 0, tempPoint, BICUBIC);

	if (finalPoint != nullptr)
	{
		for (int i = 0; i < 5; ++i)
		{
			finalPoint[i].x = tempPoint[2 * i];
			finalPoint[i].y = tempPoint[2 * i + 1];
		}
	}

	return true;
}

/**
* \brief 把 BGR 彩色通道格式转化为 YCrCb
* \param width 输入图片宽度
* \param height 输入图片高度
* \param channels 输入图片通道数
* \param BGR_data 输入数据地址
* \param YCrCb_data 输出数据地址
* \return 只有当转换成功时返回真
* \note 输入 BGR_data 中彩色图像编码必须为 BGR888 的格式
* \note YCrCb_data 应该有 BGR_data 相同的大小，转换后的 width, height, channels 与转换前完全相同
* \note channels 必须为3
* \note BGR_data 可以与 YCrCb_data 使用相同地址
* \note 此函数等价于 cv::cvtColor(BGR, YCrCb, CV_BGR2YCrCb);
* \note Y = Y, Cr = V, Cb = U
*/
bool vipl_BGR2YCrCb(int width, int height, int channels, const unsigned char *BGR_data, unsigned char *YCrCb_data)
{
	if (channels != 3) return false;
	for (int i = 0; i < width * height; ++i, BGR_data += 3, YCrCb_data += 3)
	{
		unsigned char B = BGR_data[0];
		unsigned char G = BGR_data[1];
		unsigned char R = BGR_data[2];
		unsigned int Y = (B * 1868 + G * 9617 + R * 4899 + 8192) / 16384;
		unsigned int U = ((B - Y) * 9241 + 8192) / 16384 + 128;
		unsigned int V = ((R - Y) * 11682 + 8192) / 16384 + 128;
		//		R = Y + 1.14 * V;
		//		G = Y - 0.39 * U - 0.58 * V;
		//		B = Y + 2.03 * U;
		if (Y < 0) Y = 0;
		YCrCb_data[0] = static_cast<unsigned char>(Y);
		YCrCb_data[1] = static_cast<unsigned char>(V);
		YCrCb_data[2] = static_cast<unsigned char>(U);
	}
	return true;
}

double SEETAPassiveFaceAntiSpoofing::matToTheSeetaNet(const VIPLImageData &ViplImage, const VIPLFaceInfo &face, const VIPLPoint* points5)
{
	seeta::Image frame(ViplImage.data, ViplImage.width, ViplImage.height, ViplImage.channels);

	seeta::Image crop;
	CropFace(frame, points5, crop);

	vipl_BGR2YCrCb(crop.width(), crop.height(), crop.channels(), crop.data(), crop.data());

	crop = seeta::resize(crop, seeta::Size(256, 256));

	double result = ::matToTheSeetaNet(crop, 3, 256, 256, net2timesFace, output_blob_name2timesFace);

	return result;
}


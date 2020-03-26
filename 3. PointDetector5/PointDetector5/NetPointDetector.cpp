#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "NetPointDetector.h"

VIPLNetPointDetector::VIPLNetPointDetector(const char* model_path) {
  if (model_path == nullptr) {
    input_channels_ = 1;
    input_height_ = 112;
    input_width_ = 112;
    landmark_num_ = 81;
    expand_size_ = 0.2f;
    x_move_ = 0;
    y_move_ = 0.1f;
    net_ = nullptr;
  }
  else
    LoadModel(model_path);
}

void VIPLNetPointDetector::LoadModel(const char* model_path) {
	auto bin = orz::read_file(model_path);
	if (bin.size() == 0)
	{
		std::cerr << "Error: Can not access \"" << model_path << "\"" << std::endl;
		throw std::logic_error("Missing model");
	}

	orz::FILE ifile(bin.data(), bin.size());
	auto file = &ifile;

	// FILE* file = nullptr;
  CHECK_EQ(fread(&input_channels_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&input_height_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&input_width_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&landmark_num_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&x_move_, sizeof(float), 1, file), 1);
  CHECK_EQ(fread(&y_move_, sizeof(float), 1, file), 1);
  CHECK_EQ(fread(&expand_size_, sizeof(float), 1, file), 1);
  net_ = CommonNet::Load(file);
}

void VIPLNetPointDetector::LoadModel(orz::FILE* file) {
  CHECK_EQ(fread(&input_channels_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&input_height_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&input_width_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&landmark_num_, sizeof(int), 1, file), 1);
  CHECK_EQ(fread(&x_move_, sizeof(float), 1, file), 1);
  CHECK_EQ(fread(&y_move_, sizeof(float), 1, file), 1);
  CHECK_EQ(fread(&expand_size_, sizeof(float), 1, file), 1);
  net_ = CommonNet::Load(file);
}

void VIPLNetPointDetector::ShowModelInputShape() const {
  std::cout << "<Model input shape>" << std::endl;
  std::cout << "channels:" << input_channels_ << std::endl;
  std::cout << "height:" << input_height_ << std::endl;
  std::cout << "input_width_:" << input_width_ << std::endl;
}

bool VIPLNetPointDetector::PointDetectLandmarks(const VIPLImageData &src_img, const VIPLFaceInfo face_info, VIPLPoint *landmarks, int *masks) const {
  // bounding box
  double width = face_info.width - 1, height = face_info.height - 1;
  double min_x = face_info.x, max_x = face_info.x + width;
  double min_y = face_info.y, max_y = face_info.y + height;

  // move bounding box
  min_x += width * x_move_;
  max_x += width * x_move_;
  min_y += height * y_move_;
  max_y += height * y_move_;

  //make the bounding box square
  double center_x = (min_x + max_x) / 2.0, center_y = (min_y + max_y) / 2.0;
  double r = std::max(width, height) / 2.0;
  min_x = center_x - r; max_x = center_x + r;
  min_y = center_y - r; max_y = center_y + r;
  width = max_x - min_x; height = max_y - min_y;

  // expand
  min_x = round(min_x - width * expand_size_);
  min_y = round(min_y - height * expand_size_);
  max_x = round(max_x + width * expand_size_);
  max_y = round(max_y + height * expand_size_);

  VIPLImageData dst_img(int(max_x) - int(min_x) + 1, int(max_y) - int(min_y) + 1, src_img.channels);
  std::unique_ptr<uint8_t[]> dst_img_data(new uint8_t[(int(max_x) - int(min_x) + 1) * (int(max_y) - int(min_y) + 1) * src_img.channels]);
  dst_img.data = dst_img_data.get();

  CropFace(src_img.data, src_img.width, src_img.height, src_img.channels,
    dst_img.data, int(min_x), int(min_y), int(max_x), int(max_y));

  bool flag = PredictLandmark(dst_img, landmarks, masks);

  for (int i = 0; i < landmark_num_; i++) {
    landmarks[i].x += min_x;
    landmarks[i].y += min_y;
  }

  return flag;
}

bool VIPLNetPointDetector::PredictLandmark(const VIPLImageData &src_img, VIPLPoint *landmarks, int *mask) const {
  std::vector<VIPLPoint> landmarks_vec;
  std::vector<int> masks_vec;
  if (!PredictLandmark(src_img, landmarks_vec, masks_vec)) return false;
  for (auto &ite : landmarks_vec) {
      landmarks->x = ite.x;
      landmarks->y = ite.y;
      landmarks++;
  }
  if (mask)
  {
      for (auto &ite : masks_vec) {
          *mask = ite;
          mask++;
      }
  }
  return true;
}

bool VIPLNetPointDetector::PredictLandmark(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const {
  VIPLImageData dst_img(input_width_, input_height_, input_channels_);
  std::unique_ptr<uint8_t[]> dst_img_data(new uint8_t[input_width_ * input_height_ * input_channels_]);
  dst_img.data = dst_img_data.get();

  if (
    !ResizeImage(src_img.data, src_img.width, src_img.height, src_img.channels,
    dst_img.data, input_width_, input_height_, input_channels_)
    ||
    !Predict(dst_img, landmarks, masks)
    ) return false;
  for (auto &ite : landmarks){
    ite.x *= (src_img.width - 1);
    ite.y *= (src_img.height - 1);
  }

  return true;
}

bool VIPLNetPointDetector::Predict(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const {
  if (!isLoadModel()) {
    std::cout << "<Model has not been loaded!>" << std::endl;
    return false;
  }

  if (input_channels_ != src_img.channels || input_height_ != src_img.height
    || input_width_ != src_img.width) {
    std::cout << "<Input image shape is inconsistent with model input shape!>" << std::endl;
    ShowModelInputShape();
    return false;
  }

  net_->input_blobs(0)->CopyData(1, input_height_, input_width_, input_channels_, src_img.data);
  net_->input_blobs(0)->Permute(1, 4, 2, 3);

  net_->Execute();

  float *output = net_->output_blobs(0)->data().get();
  landmarks.resize(landmark_num_);
  for (auto &ite : landmarks) {
    ite.x = *output++;
    ite.y = *output++;
  }

  masks.resize(landmark_num_, 0);
  if (net_->num_output() == 2)
  {
      float* mask_out = net_->output_blobs(1)->data().get();
      for (auto& ite : masks)
          ite = *mask_out++ > 0.5;
  }

  return true;
}

// crop src_img[min_y:max_y, min_x:max_x, :];
void VIPLNetPointDetector::CropFace(const unsigned char *src_img,
  int src_width, int src_height, int src_channels,
  unsigned char *dst_img, int min_x, int min_y, int max_x, int max_y) {
  int dst_width = max_x - min_x + 1;
  for (int r = min_y; r <= max_y; r++) {
    for (int c = min_x; c <= max_x; c++) {
      for (int ch = 0; ch < src_channels; ch++) {
        int dst_offset = ((r - min_y) * dst_width + c - min_x) * src_channels + ch;
        int src_offset = (r * src_width + c) * src_channels + ch;
        if (r < 0 || r >= src_height || c < 0 || c >= src_width) {
          dst_img[dst_offset] = 0;
        }
        else {
          dst_img[dst_offset] = src_img[src_offset];
        }
      }
    }
  }
}

/** Resize the image by bilinear interpolation & might do rgb2gray
*  @param src_im A source image
*  @param src_width The width of the source image
*  @param src_height The height of the source image
*  @param src_channels The channels of the source image, 1 or 3
*  @param[out] dst_im The target image
*  @param dst_width The width of the target image
*  @param dst_height The height of the target image
*  @param src_channels The channels of the target image, 1 or 3
*  Author: ZhangJie, HeZhenliang
*/
bool VIPLNetPointDetector::ResizeImage(
  const unsigned char *src_im, int src_width, int src_height, int src_channels,
  unsigned char *dst_im, int dst_width, int dst_height, int dst_channels) {
  if ((src_channels != 1 && src_channels != 3) || (dst_channels != 1 && dst_channels != 3)) {
    std::cout << "<Illegal image channels!>" << std::endl;
    std::cout << "src_img: " << src_channels << std::endl;
    std::cout << "dst_img: " << dst_channels << std::endl;
    return false;
  }

  if (src_width == dst_width && src_height == dst_height && src_channels == dst_channels) {
    std::memcpy(dst_im, src_im, src_width * src_height * src_channels * sizeof(unsigned char));
    return true;
  }

  double lfx_scl = double(src_width) / dst_width;
  double lfy_scl = double(src_height) / dst_height;
  double bias_x = lfx_scl / 2 - 0.5;
  double bias_y = lfy_scl / 2 - 0.5;

  std::unique_ptr<double[]> channel_buff_data(new double[src_channels]);
  double *channel_buff = channel_buff_data.get();
  for (int n_y_d = 0; n_y_d < dst_height; n_y_d++) {
    for (int n_x_d = 0; n_x_d < dst_width; n_x_d++) {
      double lf_x_s = lfx_scl * n_x_d + bias_x;
      double lf_y_s = lfy_scl * n_y_d + bias_y;

      lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
      lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
      lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
      lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

      int n_x_s = int(lf_x_s);
      int n_y_s = int(lf_y_s);

      double lf_weight_x = lf_x_s - n_x_s;
      double lf_weight_y = lf_y_s - n_y_s;

      for (int c = 0; c < src_channels; c++) {
        channel_buff[c] =
          (1 - lf_weight_y) * (1 - lf_weight_x) * src_im[(n_y_s * src_width + n_x_s) * src_channels + c] +
          (1 - lf_weight_y) * lf_weight_x * src_im[(n_y_s * src_width + n_x_s + 1) * src_channels + c] +
          lf_weight_y * (1 - lf_weight_x) * src_im[((n_y_s + 1) * src_width + n_x_s) * src_channels + c] +
          lf_weight_y * lf_weight_x * src_im[((n_y_s + 1) * src_width + n_x_s + 1) * src_channels + c];
      }
      if (src_channels <= dst_channels) {  // 1 v.s 3 or 1 v.s 1 or 3 v.s. 3
        for (int c = 0; c < dst_channels; c++) {
          dst_im[(n_y_d * dst_width + n_x_d) * dst_channels + c] =
            (unsigned char)(channel_buff[c * (src_channels == dst_channels)]);
        }
      }
      else {  // 3 v.s 1 -> Gray = 0.299 * R + 0.587 * G + 0.114 * B
        dst_im[n_y_d * dst_width + n_x_d] = (unsigned char)(
          0.299 * (channel_buff[2]) +
          0.587 * (channel_buff[1]) +
          0.114 * (channel_buff[0]));
      }
    }
  }

  return true;
}

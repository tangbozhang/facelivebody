#ifndef VIPL_NET_POINT_DETECTOR_H_
#define VIPL_NET_POINT_DETECTOR_H_

#include <memory>
#include <vector>

#include "common_net.h"
#include "VIPLStruct.h"

class VIPLNetPointDetector {
public:
  VIPLNetPointDetector(const char* model_path = nullptr);
  ~VIPLNetPointDetector() { if(net_) net_->Release(); }

  void LoadModel(const char* model_path);
  void LoadModel(orz::FILE *file);
  int LandmarkNum() const { return landmark_num_; }

  // need a cropped face
  bool PredictLandmark(const VIPLImageData &src_img, VIPLPoint *landmarks, int *masks) const;
  bool PredictLandmark(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const;

  // need an image and its face detection result(VIPLFaceInfo)
  bool PointDetectLandmarks(const VIPLImageData &src_img, VIPLFaceInfo face_info, VIPLPoint *landmarks, int *masks) const; // interface used before

private:
  int input_channels_;
  int input_height_;
  int input_width_;
  int landmark_num_;
  float x_move_;
  float y_move_;
  float expand_size_;
  std::shared_ptr<Net> net_;

  bool isLoadModel() const { return net_ != nullptr; }
  void ShowModelInputShape() const;

  // need a cropped face image with correct size 
  // corresponding to the model input size
  // output is in range [0,1]
  bool Predict(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const;

  static void CropFace(const unsigned char *src_img, int src_width, int src_height, int src_channels,
    unsigned char *dst_img, int min_x, int min_y, int max_x, int max_y);

  static bool ResizeImage(const unsigned char *src_im, int src_width, int src_height, int src_channels,
    unsigned char *dst_im, int dst_width, int dst_height, int dst_channels);
};

#endif // VIPL_NET_POINT_DETECTOR_H_

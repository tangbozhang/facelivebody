#ifndef VIPL_STABLE_POINT_DETECTOR_H_
#define VIPL_STABLE_POINT_DETECTOR_H_

#include <vector>

#include "NetPointDetector.h"

class VIPLStablePointDetector {
public:
  VIPLStablePointDetector(const char* model_path = nullptr);

  void LoadModel(const char* model_path);
  void SetStable(bool is_stable) { is_stable_ = is_stable && has_stable_model_; }
  int LandmarkNum() const;

  // need a cropped face 
  // this two functions don't use stable version 
  // because the input has been already cropped
  bool PredictLandmark(const VIPLImageData &src_img, VIPLPoint *landmarks, int *masks) const;
  bool PredictLandmark(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const;

  // need an image and its face detection result(VIPLFaceInfo)
  bool PointDetectLandmarks(const VIPLImageData &src_img, VIPLFaceInfo face_info, VIPLPoint *landmarks, int *masks) const; // interface used before
  bool PointDetectLandmarks(const VIPLImageData &src_img, VIPLFaceInfo face_info, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const; // interface used before

  // because the input has been already cropped
  bool PredictLandmark(const VIPLImageData &src_img, VIPLPoint *landmarks) const;
  bool PredictLandmark(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks) const;

  // need an image and its face detection result(VIPLFaceInfo)
  bool PointDetectLandmarks(const VIPLImageData &src_img, VIPLFaceInfo face_info, VIPLPoint *landmarks) const; // interface used before
  bool PointDetectLandmarks(const VIPLImageData &src_img, VIPLFaceInfo face_info, std::vector<VIPLPoint> &landmarks) const; // interface used before

private:
  bool is_stable_;
  bool has_stable_model_;

  VIPLNetPointDetector vipl_net_point_detector1_;
  VIPLNetPointDetector vipl_net_point_detector2_;
};

#endif // VIPL_STABLE_POINT_DETECTOR_H_
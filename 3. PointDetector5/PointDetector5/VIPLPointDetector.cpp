#include "VIPLPointDetector.h"
#include "StablePointDetector.h"

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

VIPLPointDetector::VIPLPointDetector(const char* model_path) {

	// Code
#ifdef NEED_CHECK
	checkit(model_path);
#endif

  vipl_stable_point_detector_ = std::shared_ptr<VIPLStablePointDetector>(new VIPLStablePointDetector(model_path));
}

void VIPLPointDetector::LoadModel(const char* model_path) {
  vipl_stable_point_detector_->LoadModel(model_path);
}

void VIPLPointDetector::SetStable(bool is_stable) {
  vipl_stable_point_detector_->SetStable(is_stable);
}

int VIPLPointDetector::LandmarkNum() const {
  return vipl_stable_point_detector_->LandmarkNum();
}

bool VIPLPointDetector::DetectLandmarks(const VIPLImageData &src_img, const VIPLFaceInfo& face_info, VIPLPoint *landmarks) const {
  return vipl_stable_point_detector_->PointDetectLandmarks(src_img, face_info, landmarks);
}

bool VIPLPointDetector::DetectLandmarks(const VIPLImageData& src_img, const VIPLFaceInfo& face_info, std::vector<VIPLPoint>& landmarks) const
{
	return vipl_stable_point_detector_->PointDetectLandmarks(src_img, face_info, landmarks);
}
bool VIPLPointDetector::DetectCroppedLandmarks(const VIPLImageData &src_img, VIPLPoint *landmarks) const {
  return vipl_stable_point_detector_->PredictLandmark(src_img, landmarks);
}

bool VIPLPointDetector::DetectCroppedLandmarks(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks) const {
  return vipl_stable_point_detector_->PredictLandmark(src_img, landmarks);
}

bool VIPLPointDetector::DetectCroppedLandmarks(const VIPLImageData& src_img, VIPLPoint* landmarks, int* masks) const
{
    return vipl_stable_point_detector_->PredictLandmark(src_img, landmarks, masks);
}

bool VIPLPointDetector::DetectCroppedLandmarks(const VIPLImageData& src_img, std::vector<VIPLPoint>& landmarks,
    std::vector<int>& masks) const
{
    return vipl_stable_point_detector_->PredictLandmark(src_img, landmarks, masks);
}

bool VIPLPointDetector::DetectLandmarks(const VIPLImageData& src_img, const VIPLFaceInfo& face_info,
    VIPLPoint* landmarks, int* masks) const
{
    return vipl_stable_point_detector_->PointDetectLandmarks(src_img, face_info, landmarks, masks);
}

bool VIPLPointDetector::DetectLandmarks(const VIPLImageData& src_img, const VIPLFaceInfo& face_info,
    std::vector<VIPLPoint>& landmarks, std::vector<int>& masks) const
{
    return vipl_stable_point_detector_->PointDetectLandmarks(src_img, face_info, landmarks, masks);
}


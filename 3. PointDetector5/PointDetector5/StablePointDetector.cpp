#include <limits>
#include <stdexcept>

#include "StablePointDetector.h"

#include "orz/io/i.h"

#undef STATUS
#undef DEBUG
#undef INFO
#undef ERROR

#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif

VIPLStablePointDetector::VIPLStablePointDetector(const char* model_path) {

#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif

  is_stable_ = false;
  has_stable_model_ = false;
  if (model_path != nullptr)
    LoadModel(model_path);
}

void VIPLStablePointDetector::LoadModel(const char* model_path) {

	auto bin = orz::read_file(model_path);
	if (bin.size() == 0)
	{
		std::cerr << "Error: Can not access \"" << model_path << "\"" << std::endl;
		throw std::logic_error("Missing model");
	}

#ifdef SEETA_CHECK_LOAD
	SEETA_CHECK_LOAD(bin);
#endif

	//FILE *a;
	//fopen_s(&a, model_path, "rb");
	//orz::FILE ifile(a);

	orz::FILE ifile(bin.data(), bin.size());
    auto file = &ifile;

  // FILE* file = nullptr;

  vipl_net_point_detector1_.LoadModel(file);

  if ((fread(&has_stable_model_, 1, 1, file) == 1)) {
    has_stable_model_ = true;
    fseek(file, -1, SEEK_CUR);
    vipl_net_point_detector2_.LoadModel(file);
  }
}

int VIPLStablePointDetector::LandmarkNum() const {
  if (is_stable_)
    return vipl_net_point_detector2_.LandmarkNum();
  else
    return vipl_net_point_detector1_.LandmarkNum();
}

bool VIPLStablePointDetector::PointDetectLandmarks(const VIPLImageData &src_img, const VIPLFaceInfo face_info, VIPLPoint *landmarks, int *masks) const {

#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("PointDetector");
#endif

  if (is_stable_) {
    // detect the first time
    std::unique_ptr<VIPLPoint[]> landmarks_data(new VIPLPoint[vipl_net_point_detector1_.LandmarkNum()]);
    VIPLPoint *landmarks1 = landmarks_data.get();
    if (!vipl_net_point_detector1_.PointDetectLandmarks(src_img, face_info, landmarks1, masks)) {
      return false;
    }

    // minimum enclosing rectangle
    int x_min, y_min, x_max, y_max;
    x_min = y_min = std::numeric_limits<int>::max();
    x_max = y_max = std::numeric_limits<int>::min();
    for (int i = 0; i < vipl_net_point_detector1_.LandmarkNum(); i++) {
      x_min = int(x_min < landmarks1[i].x ? x_min : landmarks1[i].x);
      y_min = int(y_min < landmarks1[i].y ? y_min : landmarks1[i].y);
      x_max = int(x_max > landmarks1[i].x ? x_max : landmarks1[i].x);
      y_max = int(y_max > landmarks1[i].y ? y_max : landmarks1[i].y);
    }

    // detect again
    VIPLFaceInfo face_info2 = { x_min, y_min, x_max - x_min + 1, y_max - y_min + 1 };
    return vipl_net_point_detector2_.PointDetectLandmarks(src_img, face_info2, landmarks, masks);
  }
  else
    return vipl_net_point_detector1_.PointDetectLandmarks(src_img, face_info, landmarks, masks);
}

bool VIPLStablePointDetector::PointDetectLandmarks(const VIPLImageData& src_img, VIPLFaceInfo face_info, std::vector<VIPLPoint>& landmarks, std::vector<int> &masks) const
{

	std::unique_ptr<VIPLPoint[]> landmarks_temp(new VIPLPoint[this->LandmarkNum()]);
	std::unique_ptr<int[]> masks_temp(new int[this->LandmarkNum()]);
    bool result = PointDetectLandmarks(src_img, face_info, landmarks_temp.get(), masks_temp.get());
	if (result)
    {
        landmarks.clear();
        landmarks.reserve(this->LandmarkNum());
        for (int i = 0; i < this->LandmarkNum(); ++i)
        {
            landmarks.push_back(landmarks_temp.get()[i]);
        }
        masks.clear();
        masks.reserve(this->LandmarkNum());
        for (int i = 0; i < this->LandmarkNum(); ++i)
        {
            masks.push_back(masks_temp.get()[i]);
        }
	}
	else
	{
		landmarks.clear();
        masks.clear();
	}
	return result;
}

bool VIPLStablePointDetector::PredictLandmark(const VIPLImageData &src_img, VIPLPoint *landmarks, int *masks) const {

#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("PointDetector");
#endif

  if (is_stable_) {
    std::cout << "Don't use this function when using stable detection!" << std::endl;
    return false;
  }
  return vipl_net_point_detector1_.PredictLandmark(src_img, landmarks, masks);
}

bool VIPLStablePointDetector::PredictLandmark(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks, std::vector<int> &masks) const {

#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("PointDetector");
#endif

  if (is_stable_) {
    std::cout << "Don't use this function when using stable detection!" << std::endl;
    return false;
  }
  return vipl_net_point_detector1_.PredictLandmark(src_img, landmarks, masks);
}

bool VIPLStablePointDetector::PredictLandmark(const VIPLImageData& src_img, VIPLPoint* landmarks) const
{
    return this->PredictLandmark(src_img, landmarks, nullptr);
}

bool VIPLStablePointDetector::PredictLandmark(const VIPLImageData& src_img, std::vector<VIPLPoint>& landmarks) const
{
    std::vector<int> masks;
    return this->PredictLandmark(src_img, landmarks, masks);
}

bool VIPLStablePointDetector::PointDetectLandmarks(const VIPLImageData& src_img, VIPLFaceInfo face_info,
    VIPLPoint* landmarks) const
{
    return this->PointDetectLandmarks(src_img, face_info, landmarks, nullptr);
}

bool VIPLStablePointDetector::PointDetectLandmarks(const VIPLImageData& src_img, VIPLFaceInfo face_info,
    std::vector<VIPLPoint>& landmarks) const
{
    std::vector<int> masks;
    return this->PointDetectLandmarks(src_img, face_info, landmarks, masks);
}

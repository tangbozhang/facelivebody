#ifndef STABLE_H
#define STABLE_H


#include "VIPLStruct.h"
#include <memory>

#include <opencv2/opencv.hpp>

class PointStable
{
public:
    PointStable(int num);
    PointStable() : PointStable(81) {}

    cv::Mat blur_image(const cv::Mat &mat);
    std::vector<VIPLPoint> deal_landmarks(const VIPLFaceInfo &face, const std::vector<VIPLPoint> &points);

private:
    bool stable = true;
    float momentum_b = 0; // 0.15
    bool blur = true; // true
    int blur_size = 5;  // 5

    std::vector<VIPLPoint> landmarks;
    std::vector<VIPLPoint> landmarks_last;

    int landmark_num;
    bool first = true;
};

#endif // STABLE_H

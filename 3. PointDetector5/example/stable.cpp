#include "stable.h"

// CopyPoint
void CopyPoint(VIPLPoint *src, VIPLPoint *dst, int num)
{
    for (int i = 0; i < num; i++)
    {
        dst[i].x = src[i].x;
        dst[i].y = src[i].y;
    }
}

void CopyPoint(VIPLPoint *src, std::vector<cv::Point2f> &dst, int num)
{
    for (int i = 0; i < num; i++)
    {
        dst[i].x = src[i].x;
        dst[i].y = src[i].y;
    }
}

void CopyPoint(std::vector<cv::Point2f> &src, VIPLPoint *dst, int num)
{
    for (int i = 0; i < num; i++)
    {
        dst[i].x = src[i].x;
        dst[i].y = src[i].y;
    }
}

// MovingAverage
float MovingAverage(float current, float last, float momentum)
{
    return current * (1 - momentum) + last * momentum;
}

VIPLPoint PointMovingAverage(VIPLPoint current, VIPLPoint last, float momentum)
{
    VIPLPoint dst;
    dst.x = MovingAverage(current.x, last.x, momentum);
    dst.y = MovingAverage(current.y, last.y, momentum);
    return dst;
}

void PointMovingAverage(VIPLPoint *current, VIPLPoint *last, VIPLPoint *dst, float momentum, int num)
{
    for (int i = 0; i < num; i++) dst[i] = PointMovingAverage(current[i], last[i], momentum);
}

void PointMovingAverage(
        const std::vector<VIPLPoint> &current,
        const std::vector<VIPLPoint> &last,
        std::vector<VIPLPoint> &dst,
        float momentum)
{
    auto num = current.size();
    for (int i = 0; i < num; i++) dst[i] = PointMovingAverage(current[i], last[i], momentum);
}

VIPLFaceInfo BBoxMovingAverage(VIPLFaceInfo current, VIPLFaceInfo last, float momentum)
{
    VIPLFaceInfo dst;
    dst.x = MovingAverage(current.x, last.x, momentum);
    dst.y = MovingAverage(current.y, last.y, momentum);
    dst.height = MovingAverage(current.height, last.height, momentum);
    dst.width = MovingAverage(current.width, last.width, momentum);
    return dst;
}

// FaceSize
float FaceSize(VIPLFaceInfo face)
{
    return std::sqrt(face.height * face.width);
}

// PointDistance
template <typename T1, typename T2>
float PointDistance(T1 a, T2 b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

PointStable::PointStable(int num)
    : landmark_num(num), landmarks(num), landmarks_last(num)
{
}

cv::Mat PointStable::blur_image(const cv::Mat &mat)
{
    auto img = mat;
    if (blur) cv::GaussianBlur(mat, img, cv::Size(blur_size, blur_size), 0, 0);
    return img;
}

std::vector<VIPLPoint> PointStable::deal_landmarks(const VIPLFaceInfo &face, const std::vector<VIPLPoint> &points)
{
    if (!stable) return points;
    landmarks = points;
    if (first || landmarks_last.size() != landmarks.size())
    {
        landmarks_last = landmarks;
    }

    int face_size = FaceSize(face);
    float mean_dist = 0;
    for (int i = 0; i < landmark_num; i++)
    {
        float dist = PointDistance(landmarks[i], landmarks_last[i]) / face_size;
        mean_dist += dist;


        float th1 = 1.0 / 100;
        float th2 = 1.0 / 50;

        if (i < 21)
        {
            th1 = 1.0 / 50;
            th2 = 1.0 / 25;
        }

        float weight;
        if (dist >= th2)
        {

            weight = 0;
        }
        else if (dist >= th1)
        {
            float k = -1.0 / (th2 - th1);
            float b = th2 / (th2 - th1);
            weight = k * dist + b;
        }
        else
        {
            weight = 1;
        }
        landmarks[i] = PointMovingAverage(landmarks[i], landmarks_last[i], weight);
        //*/
    }
    mean_dist /= landmark_num;

    // 2

    float th1 = 1.0 / 150;  // 分母越大抖动越大，滞留越小；分母越小抖动越小，滞留越大
    float th2 = 1.0 / 50;   //
    float weight;
    if (mean_dist >= th2)
    {

        weight = 0;
    }
    else if (mean_dist >= th1)
    {
        float k = -1.0 / (th2 - th1);
        float b = th2 / (th2 - th1);
        weight = k * mean_dist + b;
    }
    else
    {
        weight = 1;
    }
    PointMovingAverage(landmarks, landmarks_last, landmarks, weight);
    //*/

    // 3
    // CopyPoint(landmarks, landmarks_last, landmark_num);
    landmarks_last = landmarks;

    first = false;

    return landmarks;
}

#ifndef PUTTEXTZH_H_
#define PUTTEXTZH_H_
//#include "opencv.hpp"
#include "opencv2/opencv.hpp"
#pragma comment(lib, "Gdi32.lib")

void putTextZH(cv::Mat &dst, const char* str, cv::Point org, cv::Scalar color, int fontSize,
    const char *fn = "Arial", bool italic = false, bool underline = false);

#endif // PUTTEXT_H_
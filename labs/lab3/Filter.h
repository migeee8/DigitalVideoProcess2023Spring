#pragma once
#include <opencv2/opencv.hpp>

void myMeanBlur(cv::Mat& src, cv::Mat& dst, int kernelSize); // 自实现均值滤波
void myMedianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize); // 自实现中值滤波
void myGaussianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize, double sigma); // 自实现高斯滤波
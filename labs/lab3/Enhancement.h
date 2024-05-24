#pragma once

#include <opencv2/opencv.hpp>
void equalizeHistogram(cv::Mat& src, cv::Mat& dst, double clipLimit); // 直方图均衡化
void enhanceChromaHSI(cv::Mat& src, cv::Mat& dst, double factor); // 色相调整
void enhanceSaturationHSI(cv::Mat& src, cv::Mat& dst, double factor); // 饱和度调整
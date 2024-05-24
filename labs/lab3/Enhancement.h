#pragma once

#include <opencv2/opencv.hpp>
void equalizeHistogram(cv::Mat& src, cv::Mat& dst, double clipLimit); // ֱ��ͼ���⻯
void enhanceChromaHSI(cv::Mat& src, cv::Mat& dst, double factor); // ɫ�����
void enhanceSaturationHSI(cv::Mat& src, cv::Mat& dst, double factor); // ���Ͷȵ���
#pragma once
#include <opencv2/opencv.hpp>

void myMeanBlur(cv::Mat& src, cv::Mat& dst, int kernelSize); // ��ʵ�־�ֵ�˲�
void myMedianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize); // ��ʵ����ֵ�˲�
void myGaussianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize, double sigma); // ��ʵ�ָ�˹�˲�
#include<iostream>
#include <opencv2/opencv.hpp>
#include "Enhancement.h"

void equalizeHistogram(cv::Mat& src, cv::Mat& dst, double clipLimit) {
    if (src.channels() >= 3) {
        cv::Mat ycrcb;
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(clipLimit); // 设置对比度限制
        clahe->setTilesGridSize(cv::Size(8, 8)); // 设置块大小

        clahe->apply(channels[0], channels[0]); // 对亮度通道应用CLAHE

        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    }
    else if (src.channels() == 1) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(clipLimit); // 设置对比度限制
        clahe->setTilesGridSize(cv::Size(8, 8)); // 设置块大小

        clahe->apply(src, dst); // 对灰度图像应用CLAHE
    }
}

void enhanceChromaHSI(cv::Mat& src, cv::Mat& dst, double factor) {
    // 转换到HSI色彩空间
    cv::Mat hsi;
    cvtColor(src, hsi, cv::COLOR_BGR2HSV);

    // 分割通道
    std::vector<cv::Mat> channels;
    split(hsi, channels);

    // 增强色相分量
    channels[0] = channels[0] * factor;

    // 合并通道
    merge(channels, hsi);

    // 转换回BGR色彩空间
    cvtColor(hsi, dst, cv::COLOR_HSV2BGR);
}

void enhanceSaturationHSI(cv::Mat& src, cv::Mat& dst, double factor) {
    // 转换到HSI色彩空间
    cv::Mat hsi;
    cvtColor(src, hsi, cv::COLOR_BGR2HSV);

    // 分割通道
    std::vector<cv::Mat> channels;
    split(hsi, channels);

    // 增强饱和度分量
    channels[1] = channels[1] * factor;

    // 合并通道
    merge(channels, hsi);

    // 转换回BGR色彩空间
    cvtColor(hsi, dst, cv::COLOR_HSV2BGR);
}

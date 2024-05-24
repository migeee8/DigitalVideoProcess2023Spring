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
        clahe->setClipLimit(clipLimit); // ���öԱȶ�����
        clahe->setTilesGridSize(cv::Size(8, 8)); // ���ÿ��С

        clahe->apply(channels[0], channels[0]); // ������ͨ��Ӧ��CLAHE

        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    }
    else if (src.channels() == 1) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(clipLimit); // ���öԱȶ�����
        clahe->setTilesGridSize(cv::Size(8, 8)); // ���ÿ��С

        clahe->apply(src, dst); // �ԻҶ�ͼ��Ӧ��CLAHE
    }
}

void enhanceChromaHSI(cv::Mat& src, cv::Mat& dst, double factor) {
    // ת����HSIɫ�ʿռ�
    cv::Mat hsi;
    cvtColor(src, hsi, cv::COLOR_BGR2HSV);

    // �ָ�ͨ��
    std::vector<cv::Mat> channels;
    split(hsi, channels);

    // ��ǿɫ�����
    channels[0] = channels[0] * factor;

    // �ϲ�ͨ��
    merge(channels, hsi);

    // ת����BGRɫ�ʿռ�
    cvtColor(hsi, dst, cv::COLOR_HSV2BGR);
}

void enhanceSaturationHSI(cv::Mat& src, cv::Mat& dst, double factor) {
    // ת����HSIɫ�ʿռ�
    cv::Mat hsi;
    cvtColor(src, hsi, cv::COLOR_BGR2HSV);

    // �ָ�ͨ��
    std::vector<cv::Mat> channels;
    split(hsi, channels);

    // ��ǿ���Ͷȷ���
    channels[1] = channels[1] * factor;

    // �ϲ�ͨ��
    merge(channels, hsi);

    // ת����BGRɫ�ʿռ�
    cvtColor(hsi, dst, cv::COLOR_HSV2BGR);
}

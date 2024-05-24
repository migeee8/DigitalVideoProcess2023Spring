#include<iostream>
#include <opencv2/opencv.hpp>
#include "Filter.h"

void myMeanBlur(cv::Mat& src, cv::Mat& dst, int kernelSize) {
    if (kernelSize == 1) return;
    dst.create(src.size(), src.type());

    int border = kernelSize / 2;

    for (int y = border; y < src.rows - border; ++y) {
        for (int x = border; x < src.cols - border; ++x) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    sumB += src.at<cv::Vec3b>(y + ky, x + kx)[0];
                    sumG += src.at<cv::Vec3b>(y + ky, x + kx)[1];
                    sumR += src.at<cv::Vec3b>(y + ky, x + kx)[2];
                }
            }

            dst.at<cv::Vec3b>(y, x)[0] = sumB / (kernelSize * kernelSize);
            dst.at<cv::Vec3b>(y, x)[1] = sumG / (kernelSize * kernelSize);
            dst.at<cv::Vec3b>(y, x)[2] = sumR / (kernelSize * kernelSize);
        }
    }
}


void myMedianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize) {
    if (kernelSize == 1) return;
    dst.create(src.size(), src.type());

    int border = kernelSize / 2;

    for (int y = border; y < src.rows - border; ++y) {
        for (int x = border; x < src.cols - border; ++x) {
            std::vector<int> valuesB, valuesG, valuesR;

            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    valuesB.push_back(src.at<cv::Vec3b>(y + ky, x + kx)[0]);
                    valuesG.push_back(src.at<cv::Vec3b>(y + ky, x + kx)[1]);
                    valuesR.push_back(src.at<cv::Vec3b>(y + ky, x + kx)[2]);
                }
            }

            std::sort(valuesB.begin(), valuesB.end());
            std::sort(valuesG.begin(), valuesG.end());
            std::sort(valuesR.begin(), valuesR.end());

            dst.at<cv::Vec3b>(y, x)[0] = valuesB[valuesB.size() / 2];
            dst.at<cv::Vec3b>(y, x)[1] = valuesG[valuesG.size() / 2];
            dst.at<cv::Vec3b>(y, x)[2] = valuesR[valuesR.size() / 2];
        }
    }
}


void myGaussianBlur(cv::Mat& src, cv::Mat& dst, int kernelSize, double sigma) {
    if (kernelSize == 1) {
        dst = src.clone(); 
        return;
    }

    dst.create(src.size(), src.type());

    int border = kernelSize / 2;

    std::vector<double> kernel;

    // 生成高斯核权重
    for (int i = -border; i <= border; ++i) {
        for (int j = -border; j <= border; ++j) {
            double weight = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.push_back(weight);
        }
    }

    // 归一化权重
    double sum = 0.0;
    for (const auto& k : kernel) {
        sum += k;
    }
    for (auto& k : kernel) {
        k /= sum;
    }

    for (int y = border; y < src.rows - border; ++y) {
        for (int x = border; x < src.cols - border; ++x) {
            double sumB = 0.0, sumG = 0.0, sumR = 0.0;
            int kernelIndex = 0;

            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    int pixelB = src.at<cv::Vec3b>(y + ky, x + kx)[0];
                    int pixelG = src.at<cv::Vec3b>(y + ky, x + kx)[1];
                    int pixelR = src.at<cv::Vec3b>(y + ky, x + kx)[2];

                    sumB += pixelB * kernel[kernelIndex];
                    sumG += pixelG * kernel[kernelIndex];
                    sumR += pixelR * kernel[kernelIndex];
                    kernelIndex++;
                }
            }

            dst.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(sumB);
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(sumG);
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(sumR);
        }
    }
}



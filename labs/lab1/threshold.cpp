#include <iostream>  
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat thresholdSegmentation1(Mat frameGray) {
	//矩不变阈值分割

	// 计算灰度直方图
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	Mat hist;
	calcHist(&frameGray, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

	float totalPixels = frameGray.total();

	float minDifference = numeric_limits<float>::max();
	int optimalThreshold = 0;

	for (int t = 0; t < histSize; ++t) {
		float p0 = 0.0, p1 = 0.0;

		// 计算p0
		for (int i = 0; i <= t; ++i) {
			p0 += hist.at<float>(i) / totalPixels;
		}

		// 计算p1
		for (int i = t + 1; i < histSize; ++i) {
			p1 += hist.at<float>(i) / totalPixels;
		}

		if (p0 > 0 && p1 > 0) {

			// 计算差值
			float difference = abs(p0 - p1);

			// 更新对应最小值的阈值
			if (difference < minDifference) {
				minDifference = difference;
				optimalThreshold = t;
			}
		}
	}

	// 应用阈值分割 显示结果图像
	Mat thresholded;
	threshold(frameGray, thresholded, optimalThreshold, 255, THRESH_BINARY);
	return thresholded;
}

Mat thresholdSegmentation2(Mat frameGray) {
	//最大熵阈值分割

	//计算灰度直方图
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	Mat hist;
	calcHist(&frameGray, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

	float totalPixels = frameGray.total();
	Mat normHist = hist / totalPixels;//概率密度直方图
	Mat cumulativeHist = Mat::zeros(hist.rows, 1, CV_32F);//累计概率直方图

	float* histData = normHist.ptr<float>(0);
	float* cumHistData = cumulativeHist.ptr<float>(0);

	cumHistData[0] = histData[0]; // 初始灰度级别的累积概率等于其概率值

	for (int i = 1; i < histSize; ++i) {
		cumHistData[i] = cumHistData[i - 1] + histData[i]; // 计算累积概率
	}

	float maxSum = 0;
	int optimalThreshold = 0;
	for (int t = 0; t < histSize; ++t) {
		float e1 = 0;
		float e2 = 0;

		//计算此时前（背）景的熵
		for (int i = 0; i < t + 1; ++i)
		{
			if (cumHistData[t] < 1e-6) {
				e1 = 0;
			}
			else {
				e1 += -(histData[t] / cumHistData[t]) * log10(histData[t] / cumHistData[t]);
			}
		}

		//计算此时背（前）景的熵
		for (int j = t + 1; j < histSize; ++j)
		{
			if (1 - cumHistData[t] < 1e-6) {
				e2 = 0;
			}
			else {
				e2 += -(histData[t] /(1 -  cumHistData[t])) * log10(histData[t] / (1 - cumHistData[t]));
			}
		}

		//比较当前熵值之和和最大熵之和，如果比最大熵值要大，更新最大熵和选定阈值
		if (e1 + e2 > maxSum)
		{
			optimalThreshold = t;
			maxSum = e1 + e2;
		}
	}
	Mat thresholded;
	threshold(frameGray, thresholded, optimalThreshold, 255, THRESH_BINARY);
	return thresholded;
}

Mat thresholdSegmentation3(Mat frameGray) {
	int rows = frameGray.rows;
	int cols = frameGray.cols;
	int totalPixels = rows * cols;

	// 计算灰度直方图
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&frameGray, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

	// 大津法计算最佳阈值
	float sum = 0;
	for (int i = 0; i < histSize; ++i) {
		sum += i * hist.at<float>(i);
	}

	float sumB = 0;
	int wB = 0;
	int wF = 0;

	float varMax = 0;
	int thresh = 0;

	for (int i = 0; i < histSize; ++i) {
		wB += hist.at<float>(i);
		if (wB == 0) {
			continue;
		}
		wF = totalPixels - wB;
		if (wF == 0) {
			break;
		}

		sumB += (float)(i * hist.at<float>(i));
		float mB = sumB / wB;
		float mF = (sum - sumB) / wF;
		float varTmp = (float)wB / totalPixels * (float)wF / totalPixels * (mB - mF) * (mB - mF);

		if (varTmp > varMax) {
			varMax = varTmp;
			thresh = i;
		}
	}

	// 应用大津法计算的阈值进行分割
	Mat thresholded;
	threshold(frameGray, thresholded, thresh, 255, THRESH_BINARY);

	return thresholded;
}

Mat thresholdSegmentation4(Mat frameGray) {
	int rows = frameGray.rows;
	int cols = frameGray.cols;

	// 初始化阈值为图像最大最小灰度值的平均
	int fmax = 0, fmin = 255;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int val = static_cast<int>(frameGray.at<uchar>(i, j));
			fmax = std::max(fmax, val);
			fmin = std::min(fmin, val);
		}
	}
	int T = (fmax + fmin) / 2;

	Mat thresholded;
	while (true) {
		Mat R1, R2;
		threshold(frameGray, R1, T, 255, THRESH_BINARY);
		R2 = 255 - R1;

		// 计算区域R1和R2的均值u1和u2
		Scalar u1 = mean(frameGray, R1);
		Scalar u2 = mean(frameGray, R2);

		// 更新阈值
		int new_T = (u1[0] + u2[0]) / 2;

		// 如果阈值不再变化，则停止迭代
		if (abs(new_T - T) <= 1) {
			threshold(frameGray, thresholded, new_T, 255, THRESH_BINARY);
			break;
		}

		T = new_T;
	}

	return thresholded;
}
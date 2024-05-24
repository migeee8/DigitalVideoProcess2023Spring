#include <iostream>  
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat thresholdSegmentation1(Mat frameGray) {
	//�ز�����ֵ�ָ�

	// ����Ҷ�ֱ��ͼ
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

		// ����p0
		for (int i = 0; i <= t; ++i) {
			p0 += hist.at<float>(i) / totalPixels;
		}

		// ����p1
		for (int i = t + 1; i < histSize; ++i) {
			p1 += hist.at<float>(i) / totalPixels;
		}

		if (p0 > 0 && p1 > 0) {

			// �����ֵ
			float difference = abs(p0 - p1);

			// ���¶�Ӧ��Сֵ����ֵ
			if (difference < minDifference) {
				minDifference = difference;
				optimalThreshold = t;
			}
		}
	}

	// Ӧ����ֵ�ָ� ��ʾ���ͼ��
	Mat thresholded;
	threshold(frameGray, thresholded, optimalThreshold, 255, THRESH_BINARY);
	return thresholded;
}

Mat thresholdSegmentation2(Mat frameGray) {
	//�������ֵ�ָ�

	//����Ҷ�ֱ��ͼ
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	Mat hist;
	calcHist(&frameGray, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

	float totalPixels = frameGray.total();
	Mat normHist = hist / totalPixels;//�����ܶ�ֱ��ͼ
	Mat cumulativeHist = Mat::zeros(hist.rows, 1, CV_32F);//�ۼƸ���ֱ��ͼ

	float* histData = normHist.ptr<float>(0);
	float* cumHistData = cumulativeHist.ptr<float>(0);

	cumHistData[0] = histData[0]; // ��ʼ�Ҷȼ�����ۻ����ʵ��������ֵ

	for (int i = 1; i < histSize; ++i) {
		cumHistData[i] = cumHistData[i - 1] + histData[i]; // �����ۻ�����
	}

	float maxSum = 0;
	int optimalThreshold = 0;
	for (int t = 0; t < histSize; ++t) {
		float e1 = 0;
		float e2 = 0;

		//�����ʱǰ������������
		for (int i = 0; i < t + 1; ++i)
		{
			if (cumHistData[t] < 1e-6) {
				e1 = 0;
			}
			else {
				e1 += -(histData[t] / cumHistData[t]) * log10(histData[t] / cumHistData[t]);
			}
		}

		//�����ʱ����ǰ��������
		for (int j = t + 1; j < histSize; ++j)
		{
			if (1 - cumHistData[t] < 1e-6) {
				e2 = 0;
			}
			else {
				e2 += -(histData[t] /(1 -  cumHistData[t])) * log10(histData[t] / (1 - cumHistData[t]));
			}
		}

		//�Ƚϵ�ǰ��ֵ֮�ͺ������֮�ͣ�����������ֵҪ�󣬸�������غ�ѡ����ֵ
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

	// ����Ҷ�ֱ��ͼ
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&frameGray, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

	// ��򷨼��������ֵ
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

	// Ӧ�ô�򷨼������ֵ���зָ�
	Mat thresholded;
	threshold(frameGray, thresholded, thresh, 255, THRESH_BINARY);

	return thresholded;
}

Mat thresholdSegmentation4(Mat frameGray) {
	int rows = frameGray.rows;
	int cols = frameGray.cols;

	// ��ʼ����ֵΪͼ�������С�Ҷ�ֵ��ƽ��
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

		// ��������R1��R2�ľ�ֵu1��u2
		Scalar u1 = mean(frameGray, R1);
		Scalar u2 = mean(frameGray, R2);

		// ������ֵ
		int new_T = (u1[0] + u2[0]) / 2;

		// �����ֵ���ٱ仯����ֹͣ����
		if (abs(new_T - T) <= 1) {
			threshold(frameGray, thresholded, new_T, 255, THRESH_BINARY);
			break;
		}

		T = new_T;
	}

	return thresholded;
}
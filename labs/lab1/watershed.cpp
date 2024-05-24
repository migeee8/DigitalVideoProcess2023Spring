#include <iostream>  
#include <opencv2/opencv.hpp>
#include "threshold.h"

using namespace cv;
using namespace std;

Vec3b RandomColor(int value)
{
	value = value % 255;
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

Mat watershedSegmentation1(Mat frame) {

	//预处理
	Mat frameGray, frameBlur, frameCanny;
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);
	GaussianBlur(frameGray, frameBlur, Size(5, 5), 2);
	Canny(frameBlur, frameCanny, 80, 150);
	//imshow("show", frameGray);

	//寻找轮廓当分水岭函数的种子点
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(frameCanny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	Mat marks = Mat::zeros(frame.size(), CV_32S);
	for (int i = 0; i < contours.size(); i++) {
		drawContours(marks, contours, i, Scalar::all(i + 1), 1, 8, hierarchy);
	}

	//分水岭函数
	watershed(frame, marks);
	
	//Mat showMarkers;
	//normalize(marks, showMarkers, 0, 255, NORM_MINMAX, CV_8U);
	//imshow("show", showMarkers);

	//分区域上色
	Mat PerspectiveImage = Mat::zeros(frame.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++) {
		for (int j = 0; j < marks.cols; j++) {
			int index = marks.at<int>(i, j);
			if (index == -1) {
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else {
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	Mat wshed;
	addWeighted(frame, 0.4, PerspectiveImage, 0.6, 0, wshed);
	/*imshow("show", wshed);*/
	return wshed;
}

Mat watershedSegmentation2(Mat frame) {
	Mat frameGray;
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);

	Mat frameBlur, gradient;
	Mat sobelX, sobelY;
	GaussianBlur(frameGray, frameBlur, Size(9, 9), 2);   //高斯滤波
	GaussianBlur(frameBlur, frameBlur, Size(5, 5), 2);   //高斯滤波
	//imshow("gauss", frameBlur);

	//梯度图
	Sobel(frameBlur, sobelX, CV_32F, 1, 0);
	Sobel(frameBlur, sobelY, CV_32F, 0, 1);
	magnitude(sobelX, sobelY, gradient);
	normalize(gradient, gradient, 0, 255, NORM_MINMAX, CV_8U);
	//addWeighted(sobelX, 0.5, sobelY, 0.5, 0, gradient);
	//imshow("梯度图", gradient);
	//cv::waitKey(0);
	vector<pair<int, Point>> pixels; // <灰度值， 所在点的坐标>

	// 所有点按照上面的pair结构存储在pixels里
	for (int y = 0; y < frameGray.rows; ++y) {
		for (int x = 0; x < frameGray.cols; ++x) {
			//cout << "gradient: " << static_cast<int>(gradient.at<uchar>(y, x)) << " x: " << x << " y: " << y << endl;
			pixels.push_back(make_pair(static_cast<int>(gradient.at<uchar>(y, x)), Point(x, y)));
		}
	}

	// 基于灰度值排序
	sort(pixels.begin(), pixels.end(), [](const pair<int, Point>& a, const pair<int, Point>& b) {
		return a.first < b.first;
		});

	Mat markers = Mat::zeros(frameGray.size(), CV_32SC1);
	int currentMarker = 0;

	for (const auto& pixel : pixels) {
		int x = pixel.second.x;
		int y = pixel.second.y;

		if (markers.at<int>(y, x) == 0) {
			queue<Point> q;
			q.push(Point(x, y));
			++currentMarker;

			while (!q.empty()) {
				Point p = q.front();
				q.pop();

				//遍历该点周围的点
				for (int ny = p.y - 1; ny <= p.y + 1; ++ny) {
					for (int nx = p.x - 1; nx <= p.x + 1; ++nx) {
						if (ny >= 0 && ny < frameGray.rows && nx >= 0 && nx < frameGray.cols) {
							//如果是没有标记的点，且和当前的灰度差小于特定阈值
							if (markers.at<int>(ny, nx) == 0 && abs(gradient.at<uchar>(ny, nx) - gradient.at<uchar>(p)) < 4) {
								markers.at<int>(ny, nx) = currentMarker; //标记
								q.push(Point(nx, ny));//加入区域的队列
							}
						}
					}
				}
			}
		}
	}

	//分区域着色
	vector<Vec3b> colors(currentMarker + 1);
	for (int i = 0; i <= currentMarker; ++i) {
		colors[i] = Vec3b(rand() & 255, rand() & 255, rand() & 255);
	}

	Mat segmented(frameGray.size(), CV_8UC3);
	for (int y = 0; y < frameGray.rows; ++y) {
		for (int x = 0; x < frameGray.cols; ++x) {
			int index = markers.at<int>(y, x);
			segmented.at<Vec3b>(y, x) = colors[index];
		}
	}
	//imshow("show1", segmented);
	Mat wshed;
	addWeighted(frame, 0.4, segmented, 0.6, 0, wshed);
	//imshow("show2", wshed);
	return wshed;
}

Mat watershedSegmentation3(Mat frame) {
	Mat frameGray;
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);

	Mat frameBlur, gradient;
	Mat sobelX, sobelY;
	GaussianBlur(frameGray, frameBlur, Size(9, 9), 2);   //高斯滤波
	GaussianBlur(frameBlur, frameBlur, Size(5, 5), 2);   //高斯滤波
	//imshow("gray", frameGray);
	//imshow("gradient", gradient);
	//cv::waitKey(0);

	Sobel(frameBlur, sobelX, CV_16S, 1, 0, 3);
	convertScaleAbs(sobelX, sobelX);
	//imshow("边缘图x", sobelX);
	//cv ::waitKey(0);
	Sobel(frameBlur, sobelY, CV_16S, 0, 1, 3);
	convertScaleAbs(sobelY, sobelY);
	//imshow("边缘图y", sobelY);
	//cv::waitKey(0);

	addWeighted(sobelX, 0.5, sobelY, 0.5, 0, gradient);
	//imshow("梯度图", gradient);
	//cv::waitKey(0);

	vector<pair<int, Point>> pixels; // <灰度值， 所在点的坐标>

	// 所有点按照上面的pair结构存储在pixels里
	for (int y = 0; y < frameGray.rows; ++y) {
		for (int x = 0; x < frameGray.cols; ++x) {
			//cout << "gradient: " << static_cast<int>(gradient.at<uchar>(y, x)) << " x: " << x << " y: " << y << endl;
			pixels.push_back(make_pair(static_cast<int>(gradient.at<uchar>(y, x)), Point(x, y)));
		}
	}

	// 基于灰度值排序
	sort(pixels.begin(), pixels.end(), [](const pair<int, Point>& a, const pair<int, Point>& b) {
		return a.first < b.first; 
		});

	Mat markers = Mat::zeros(frameGray.size(), CV_32SC1);
	int currentMarker = 0;

	for (const auto& pixel : pixels) {
		int x = pixel.second.x;
		int y = pixel.second.y;

		if (markers.at<int>(y, x) == 0) {
			queue<Point> q;
			q.push(Point(x, y));
			++currentMarker;

			while (!q.empty()) {
				Point p = q.front();
				q.pop();

				//遍历该点周围的点
				for (int ny = p.y - 1; ny <= p.y + 1; ++ny) {
					for (int nx = p.x - 1; nx <= p.x + 1; ++nx) {
						if (ny >= 0 && ny < frameGray.rows && nx >= 0 && nx < frameGray.cols) {
							//如果是没有标记的点，且和当前的灰度差小于特定阈值
							if (markers.at<int>(ny, nx) == 0 && abs(gradient.at<uchar>(ny, nx) - gradient.at<uchar>(p)) < 4) {
								markers.at<int>(ny, nx) = currentMarker; //标记
								q.push(Point(nx, ny));//加入区域的队列
							}
						}
					}
				}
			}
		}
	}

	vector<Vec3b> colors(currentMarker + 1);
	for (int i = 0; i <= currentMarker; ++i) {
		colors[i] = Vec3b(rand() & 255, rand() & 255, rand() & 255);
	}

	Mat segmented(frameGray.size(), CV_8UC3);
	for (int y = 0; y < frameGray.rows; ++y) {
		for (int x = 0; x < frameGray.cols; ++x) {
			int index = markers.at<int>(y, x);
			segmented.at<Vec3b>(y, x) = colors[index];
		}
	}

	Mat wshed;
	addWeighted(frame, 0.4, segmented, 0.6, 0, wshed);
	//imshow("segmented", segmented);
	//imshow("wshed", wshed);
	return wshed;
}

Mat watershedSegmentation4(Mat frame) {
	//二值化图像
	Mat binary, gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	//imshow("original", frame);
	//imshow("gray", gray);

	//高斯模糊
	Mat Gauss;
	GaussianBlur(gray, Gauss, Size(5, 5), 10.0);
	//imshow("blur", Gauss);

	//二值化
	binary = thresholdSegmentation3(Gauss);
	//imshow("bi", binary);

	
	vector<vector<Point>> contours;
	findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat markers = Mat::zeros(frame.size(), CV_32SC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	}

	//Mat showMarkers;
	//normalize(markers, showMarkers, 0, 255, NORM_MINMAX, CV_8U);
	//imshow("markers", showMarkers);


	//分割区域着色
	vector<Vec3b> colors(contours.size() + 1);
	for (int i = 0; i <= contours.size(); ++i) {
		colors[i] = Vec3b(rand() & 255, rand() & 255, rand() & 255);
	}

	Mat segmented(frame.size(), CV_8UC3);
	for (int y = 0; y < frame.rows; ++y) {
		for (int x = 0; x < frame.cols; ++x) {
			int index = markers.at<int>(y, x);
			segmented.at<Vec3b>(y, x) = colors[index];
		}
	}

	Mat wshed;
	addWeighted(frame, 0.4, segmented, 0.6, 1, wshed);
	//imshow("seg", segmented);
	//imshow("wshed", wshed);
	return wshed;
}
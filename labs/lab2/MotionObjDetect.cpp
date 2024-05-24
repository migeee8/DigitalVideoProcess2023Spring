#include "stdio.h"
#include <iostream>  
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void moveCheck1(Mat preframe, Mat frame) {
    //灰度图
    Mat preGray, gray;
    cvtColor(preframe, preGray, COLOR_BGR2GRAY);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    //imshow("show", gray);

    //计算帧差图
    Mat diff;
    absdiff(preGray, gray, diff);
    //imshow("show", diff);

    //二值化
    int thresholdValue = 25;
    Mat binary;
    threshold(diff, binary, thresholdValue, 255, THRESH_BINARY);
    //imshow("show", binary);


    //腐蚀 去掉噪点
    Mat eroElement = getStructuringElement(MORPH_RECT, Size(1, 5));
    erode(binary, binary, eroElement);
    //imshow("ero", binary);


    //膨胀 将移动物体用物块表示
    Mat dilElement = getStructuringElement(MORPH_RECT, Size(5, 30));
    dilate(binary, binary, dilElement);
    //imshow("bi1", binary);

    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

    Mat res = frame.clone();

    for (size_t c = 0; c < contours.size(); ++c) {
        double area = cv::contourArea(contours[c]);
        if (area < 500) {
            continue;
        }
        Rect boundBox = boundingRect(contours[c]);
        // 绘制轮廓
        drawContours(res, vector<vector<Point>>{contours[c]}, 0, Scalar(0, 0, 255), 2);
        // 绘制外接矩形
        rectangle(res, boundBox, Scalar(0, 255, 0), 2);
    }

    // 显示图像
    imshow("帧差法", res);
    waitKey(10);
}

void moveCheck2(Mat frame, Ptr<BackgroundSubtractor> pMOG2) {
    // 使用 MOG2 进行背景减除
    
    Mat fgMask;
    pMOG2->apply(frame, fgMask, 0.005);
    //imshow("show", fgMask);

    // 开操作，去除噪点
    Mat line = getStructuringElement(MORPH_RECT, Size(1, 5), Point(-1, -1));
    morphologyEx(fgMask, fgMask, MORPH_OPEN, line);
    imshow("subtracted", fgMask);

    // 执行轮廓检测
    vector<vector<Point>> contours;
    findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat res = frame.clone();

    for (size_t c = 0; c < contours.size(); ++c) {
        double area = cv::contourArea(contours[c]);
        if (area < 150) {
            continue;
        }
        Rect boundBox = boundingRect(contours[c]);
        // 绘制轮廓
        drawContours(res, vector<vector<Point>>{contours[c]}, 0, Scalar(0, 0, 255), 2);
        // 绘制外接矩形
        rectangle(res, boundBox, Scalar(0, 255, 0), 2);
    }

    // 显示结果
    imshow("背景减法", res);
}


int main() 
{
	string path = "D://course//DigitalVideoProcess//labs//lab2//video//exp2.avi"; 

	VideoCapture cap(path);
	if (!cap.isOpened()) {
		cout << "Fail opening the video!";
		return -1;
	}

	Mat preframe,frame;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(500, 100, false);

	while (cap.read(frame))
	{
		imshow("video", frame);

		if (!preframe.empty())
		{
            moveCheck2(frame, pMOG2);//背景减法
            moveCheck1(preframe, frame);//帧差法
		}
		
		frame.copyTo(preframe);//更新前一帧的参数

		if (waitKey(10) == 'q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}


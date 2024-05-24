#include "stdio.h"
#include <iostream>  
#include <opencv2/opencv.hpp>
#include "watershed.h"
#include "threshold.h"

using namespace cv;
using namespace std;

#define THRESHOLD_VALUE 20
#define DIFF_P 0.6

void showTheImage(Mat mat, const char* name, int height, int width, int left, int top)
{
	namedWindow(name, 0);
	resizeWindow(name, width, height);
	moveWindow(name, left, top);
	imshow(name, mat);
}
	
int main()
{
	string path = "D://course//DigitalVideoProcess//labs//lab1//video//exp1.avi";

	VideoCapture cap(path);
	if (!cap.isOpened()) {
		cout << "Fail opening the video!";
		return -1;
	}

	Mat preframeGray, frame;//输入
	Mat frameGray, diff;//处理过程
	Mat threshold1, threshold2, threshold3, threshold4, watershed1, watershed2, watershed3, watershed4;//输出

	cap.read(frame);
	int pixelNum = frame.total();
	cout << pixelNum;
	cap.set(CAP_PROP_POS_FRAMES, 0);

	while (cap.read(frame)) {

		showTheImage(frame, "Original video", 240, 320, 50, 20);

		cvtColor(frame, frameGray, COLOR_BGR2GRAY);//当前帧转为灰度图

		if (!preframeGray.empty())
		{
			absdiff(preframeGray, frameGray, diff);//计算和前一帧的difference矩阵

			Mat thresholdedDiff;
			threshold(diff, thresholdedDiff, THRESHOLD_VALUE, 255, THRESH_BINARY);

			int nonZeroCount = countNonZero(thresholdedDiff); // 变化的像素值数量

			float p = 1.0 * nonZeroCount / pixelNum ;// 计算变化的像素占比 如果该比例大于设定的比值，那么将该帧设置为关键帧

			if (p > DIFF_P)//如果是关键帧，①显示②做分割处理
			{
				std::cout << nonZeroCount << " " << p << endl;
				
				threshold1 = thresholdSegmentation1(frameGray); //阈值分割 矩不变
				threshold2 = thresholdSegmentation2(frameGray);//阈值分割 最大熵值
				threshold3 = thresholdSegmentation3(frameGray);//阈值分割 大津法
				threshold4 = thresholdSegmentation4(frameGray);//阈值分割 迭代法
				watershed1 = watershedSegmentation1(frame);//分水岭 0
				watershed2 = watershedSegmentation2(frame);//分水岭 1
				watershed3 = watershedSegmentation3(frame);//分水岭 2
				watershed4 = watershedSegmentation4(frame);//分水岭 3

				showTheImage(frame, "关键帧", 240, 320, 400, 20);
				showTheImage(threshold1, "矩不变阈值分割", 240, 320, 50, 300);
				showTheImage(threshold2, "最大熵阈值分割", 240, 320, 400, 300);
				showTheImage(threshold3, "大津法阈值分割", 240, 320, 50, 580);
				showTheImage(threshold4, "迭代法阈值分割", 240, 320, 400, 580);
				showTheImage(watershed1, "分水岭0", 240, 320, 750, 20);
				showTheImage(watershed2, "分水岭1", 240, 320, 750, 300);
				showTheImage(watershed3, "分水岭2", 240, 320, 1100, 20);
				showTheImage(watershed4, "分水岭3", 240, 320, 1100, 300);
			}
		}
		else //对第一帧图像 直接当作关键帧，①显示②做分割处理
		{

			threshold1 = thresholdSegmentation1(frameGray); //阈值分割 矩不变
			threshold2 = thresholdSegmentation2(frameGray);//阈值分割 最大熵值
			threshold3 = thresholdSegmentation3(frameGray);//阈值分割 大津法
			threshold4 = thresholdSegmentation4(frameGray);//阈值分割 迭代法
			watershed1 = watershedSegmentation1(frame);//分水岭 0
			watershed2 = watershedSegmentation2(frame);//分水岭 1
			watershed3 = watershedSegmentation3(frame);//分水岭 2
			watershed4 = watershedSegmentation4(frame);//分水岭 3

			showTheImage(frame, "关键帧", 240, 320, 400, 20);
			showTheImage(threshold1, "矩不变阈值分割", 240, 320, 50, 300);
			showTheImage(threshold2, "最大熵阈值分割", 240, 320, 400, 300);
			showTheImage(threshold3, "大津法阈值分割", 240, 320, 50, 580);
			showTheImage(threshold4, "迭代法阈值分割", 240, 320, 400, 580);
			showTheImage(watershed1, "分水岭0", 240, 320, 750, 20);
			showTheImage(watershed2, "分水岭1", 240, 320, 750, 300);
			showTheImage(watershed3, "分水岭2", 240, 320, 1100, 20);
			showTheImage(watershed4, "分水岭3", 240, 320, 1100, 300);
		}

		frameGray.copyTo(preframeGray);//更新前一帧的参数

		if (waitKey(25) == 'q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}
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

	Mat preframeGray, frame;//����
	Mat frameGray, diff;//�������
	Mat threshold1, threshold2, threshold3, threshold4, watershed1, watershed2, watershed3, watershed4;//���

	cap.read(frame);
	int pixelNum = frame.total();
	cout << pixelNum;
	cap.set(CAP_PROP_POS_FRAMES, 0);

	while (cap.read(frame)) {

		showTheImage(frame, "Original video", 240, 320, 50, 20);

		cvtColor(frame, frameGray, COLOR_BGR2GRAY);//��ǰ֡תΪ�Ҷ�ͼ

		if (!preframeGray.empty())
		{
			absdiff(preframeGray, frameGray, diff);//�����ǰһ֡��difference����

			Mat thresholdedDiff;
			threshold(diff, thresholdedDiff, THRESHOLD_VALUE, 255, THRESH_BINARY);

			int nonZeroCount = countNonZero(thresholdedDiff); // �仯������ֵ����

			float p = 1.0 * nonZeroCount / pixelNum ;// ����仯������ռ�� ����ñ��������趨�ı�ֵ����ô����֡����Ϊ�ؼ�֡

			if (p > DIFF_P)//����ǹؼ�֡������ʾ�����ָ��
			{
				std::cout << nonZeroCount << " " << p << endl;
				
				threshold1 = thresholdSegmentation1(frameGray); //��ֵ�ָ� �ز���
				threshold2 = thresholdSegmentation2(frameGray);//��ֵ�ָ� �����ֵ
				threshold3 = thresholdSegmentation3(frameGray);//��ֵ�ָ� ���
				threshold4 = thresholdSegmentation4(frameGray);//��ֵ�ָ� ������
				watershed1 = watershedSegmentation1(frame);//��ˮ�� 0
				watershed2 = watershedSegmentation2(frame);//��ˮ�� 1
				watershed3 = watershedSegmentation3(frame);//��ˮ�� 2
				watershed4 = watershedSegmentation4(frame);//��ˮ�� 3

				showTheImage(frame, "�ؼ�֡", 240, 320, 400, 20);
				showTheImage(threshold1, "�ز�����ֵ�ָ�", 240, 320, 50, 300);
				showTheImage(threshold2, "�������ֵ�ָ�", 240, 320, 400, 300);
				showTheImage(threshold3, "�����ֵ�ָ�", 240, 320, 50, 580);
				showTheImage(threshold4, "��������ֵ�ָ�", 240, 320, 400, 580);
				showTheImage(watershed1, "��ˮ��0", 240, 320, 750, 20);
				showTheImage(watershed2, "��ˮ��1", 240, 320, 750, 300);
				showTheImage(watershed3, "��ˮ��2", 240, 320, 1100, 20);
				showTheImage(watershed4, "��ˮ��3", 240, 320, 1100, 300);
			}
		}
		else //�Ե�һ֡ͼ�� ֱ�ӵ����ؼ�֡������ʾ�����ָ��
		{

			threshold1 = thresholdSegmentation1(frameGray); //��ֵ�ָ� �ز���
			threshold2 = thresholdSegmentation2(frameGray);//��ֵ�ָ� �����ֵ
			threshold3 = thresholdSegmentation3(frameGray);//��ֵ�ָ� ���
			threshold4 = thresholdSegmentation4(frameGray);//��ֵ�ָ� ������
			watershed1 = watershedSegmentation1(frame);//��ˮ�� 0
			watershed2 = watershedSegmentation2(frame);//��ˮ�� 1
			watershed3 = watershedSegmentation3(frame);//��ˮ�� 2
			watershed4 = watershedSegmentation4(frame);//��ˮ�� 3

			showTheImage(frame, "�ؼ�֡", 240, 320, 400, 20);
			showTheImage(threshold1, "�ز�����ֵ�ָ�", 240, 320, 50, 300);
			showTheImage(threshold2, "�������ֵ�ָ�", 240, 320, 400, 300);
			showTheImage(threshold3, "�����ֵ�ָ�", 240, 320, 50, 580);
			showTheImage(threshold4, "��������ֵ�ָ�", 240, 320, 400, 580);
			showTheImage(watershed1, "��ˮ��0", 240, 320, 750, 20);
			showTheImage(watershed2, "��ˮ��1", 240, 320, 750, 300);
			showTheImage(watershed3, "��ˮ��2", 240, 320, 1100, 20);
			showTheImage(watershed4, "��ˮ��3", 240, 320, 1100, 300);
		}

		frameGray.copyTo(preframeGray);//����ǰһ֡�Ĳ���

		if (waitKey(25) == 'q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}
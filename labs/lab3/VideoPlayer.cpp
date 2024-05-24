#include <opencv2/opencv.hpp>
#include <iostream>

#include "Enhancement.h"
#include "Filter.h"

using namespace cv;
using namespace std;

VideoCapture cap;
Mat frame;
bool isPlaying = true;
double fps; // ֡��
int totalFrames; // ��֡��
int currentFrame = 0; // ��ǰ֡������

/* ���������ǿ���� */
double brightness = 0.0;
double contrast = 1.0;
double clipLimit = 1.0;
double HUEfactor = 1.0;
double saturation = 1.0;
int kernelSizeMedian = 1;
int kernelSizeMean = 1;
int gaussKernelSize = 1;

 /* �������ص����� */
void onTrackbarProgress(int pos, void* userdata) {
    cap.set(CAP_PROP_POS_FRAMES, pos);
    currentFrame = pos;
}

void onTrackbarBright(int pos, void* userdata) {
    brightness = pos - 50; // ���Ȳ���ӳ�䵽[-50, 50], ����0������С��0�䰵
}

void  onTrackbarContrast(int pos, void* userdata) {
    contrast = pos / 100.0;// �ԱȶȲ���ӳ�䵽[0, 3]������1�Ա���ǿ��С��1�Աȼ���
}

void  onTrackbarHistEq(int pos, void* userdata) {
    clipLimit = pos / 10.0;// ֱ��ͼ���⻯����ӳ�䵽[0, 10]
}

void  onTrackbarHUE(int pos, void* userdata) {
    HUEfactor = pos / 100.0;// HSIɫ�Ȳ���HUEӳ�䵽[0, 2]
}

void onTrackbarSaturation(int pos, void* userdata) {
    saturation = pos / 100.0;// ���Ͷ�
}

void onTrackbarMeanKernelSize(int pos, void* userdata) {
    if (pos % 2 == 0) {
        pos++; // �˴�СΪ����
    }
    gaussKernelSize = pos;
}

void onTrackbarMedianKernelSize(int pos, void* userdata) {
    if (pos % 2 == 0) {
        pos++; 
    }
    gaussKernelSize = pos;
}

void onTrackbarGaussianKernelSize(int pos, void* userdata) {
    if (pos % 2 == 0) {
        pos++;
    }
    gaussKernelSize = pos;
}

void Adjust(Mat& src, Mat& dst) {
    src.convertTo(dst, -1, contrast, brightness);
    equalizeHistogram(dst, dst, clipLimit);
    enhanceChromaHSI(dst, dst, HUEfactor);
    enhanceSaturationHSI(dst, dst, saturation);
    //medianBlur(dst, dst, kernelSizeMedian);
    myMeanBlur(dst, dst, kernelSizeMedian);
    //blur(dst, dst, Size(kernelSizeMean, kernelSizeMean));
    myMedianBlur(dst, dst, kernelSizeMean);
    GaussianBlur(dst, dst, Size(gaussKernelSize, gaussKernelSize), 0, 0);
    //myGaussianBlur(dst, dst, gaussKernelSize, 1);
}

int main() {
    string path = "D://course//DigitalVideoProcess//labs//lab3//video//exp3.avi";
    cap.open(path);

    if (!cap.isOpened()) {
        cout << "Error opening video file" << endl;
        return -1;
    }

    /*��Ƶ���Ŵ���*/
    namedWindow("Video Player", WINDOW_NORMAL);
    resizeWindow("Video Player", 800, 600); 
    fps = cap.get(CAP_PROP_FPS);
    totalFrames = cap.get(CAP_PROP_FRAME_COUNT);

    createTrackbar("Progress", "Video Player", &currentFrame, totalFrames, onTrackbarProgress);


    /*�������ڴ���*/
    namedWindow("Adjust", WINDOW_NORMAL);
    resizeWindow("Adjust", 400, 200); 

    int brightnessBar = 50;
    createTrackbar("����", "Adjust", &brightnessBar, 100, onTrackbarBright);
    setTrackbarPos("����", "Adjust", brightnessBar);

    int contrastBar = 100;
    createTrackbar("�Աȶ�", "Adjust", &contrastBar, 300, onTrackbarContrast);
    setTrackbarPos("�Աȶ�", "Adjust", contrastBar);

    int HistEqBar = 10;
    createTrackbar("ֱ��ͼ����", "Adjust", &HistEqBar, 100, onTrackbarHistEq);
    setTrackbarPos("ֱ��ͼ����", "Adjust", HistEqBar);

    int HueBar = 100;
    createTrackbar("ɫ��", "Adjust", &HueBar, 200, onTrackbarHUE);
    setTrackbarPos("ɫ��", "Adjust", HueBar);

    int SaturationBar = 100;
    createTrackbar("���Ͷ�", "Adjust", &SaturationBar, 200, onTrackbarSaturation);
    setTrackbarPos("���Ͷ�", "Adjust", SaturationBar);

    int meanBar = 1;
    createTrackbar("��ֵ�˲���", "Adjust", &meanBar, 31, onTrackbarMeanKernelSize);
    setTrackbarPos("��ֵ�˲���", "Adjust", meanBar);

    int medianBar = 1;
    createTrackbar("��ֵ�˲���", "Adjust", &medianBar, 31, onTrackbarMedianKernelSize);
    setTrackbarPos("��ֵ�˲���", "Adjust", medianBar);

    int gaussianBar = 1;
    createTrackbar("��˹�˲���", "Adjust", &gaussianBar, 31, onTrackbarGaussianKernelSize);
    setTrackbarPos("��˹�˲���", "Adjust", gaussianBar);


    while (true) {
        if (isPlaying) {
            cap >> frame;

            if (frame.empty()) {
                cout << "End of video" << endl;
                break;
            }

            Mat adjustedFrame;
            Adjust(frame, adjustedFrame);

            imshow("Video Player", adjustedFrame);
            currentFrame = cap.get(CAP_PROP_POS_FRAMES);
            setTrackbarPos("Progress", "Video Player", currentFrame);

            int key = waitKey(1000 / fps); 

            if (key == ' ') { // ���ո����ͣ�򲥷�
                isPlaying = !isPlaying;
            }
        }
        else {
            Mat adjustedFrame;
            Adjust(frame, adjustedFrame);

            Mat pausedFrame = adjustedFrame.clone();
            putText(pausedFrame, "PAUSED", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            imshow("Video Player", pausedFrame);

            int key = waitKey(0);
            if (key == ' ') { // ���ո����ͣ�򲥷�
                isPlaying = !isPlaying;
            }
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
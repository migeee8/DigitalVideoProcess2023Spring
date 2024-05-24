#include <opencv2/opencv.hpp>
#include <iostream>

#include "Enhancement.h"
#include "Filter.h"

using namespace cv;
using namespace std;

VideoCapture cap;
Mat frame;
bool isPlaying = true;
double fps; // 帧率
int totalFrames; // 总帧数
int currentFrame = 0; // 当前帧计数器

/* 画面调整增强参数 */
double brightness = 0.0;
double contrast = 1.0;
double clipLimit = 1.0;
double HUEfactor = 1.0;
double saturation = 1.0;
int kernelSizeMedian = 1;
int kernelSizeMean = 1;
int gaussKernelSize = 1;

 /* 滑动条回调函数 */
void onTrackbarProgress(int pos, void* userdata) {
    cap.set(CAP_PROP_POS_FRAMES, pos);
    currentFrame = pos;
}

void onTrackbarBright(int pos, void* userdata) {
    brightness = pos - 50; // 亮度参数映射到[-50, 50], 大于0变亮，小于0变暗
}

void  onTrackbarContrast(int pos, void* userdata) {
    contrast = pos / 100.0;// 对比度参数映射到[0, 3]，大于1对比增强，小于1对比减弱
}

void  onTrackbarHistEq(int pos, void* userdata) {
    clipLimit = pos / 10.0;// 直方图均衡化参数映射到[0, 10]
}

void  onTrackbarHUE(int pos, void* userdata) {
    HUEfactor = pos / 100.0;// HSI色度参数HUE映射到[0, 2]
}

void onTrackbarSaturation(int pos, void* userdata) {
    saturation = pos / 100.0;// 饱和度
}

void onTrackbarMeanKernelSize(int pos, void* userdata) {
    if (pos % 2 == 0) {
        pos++; // 核大小为奇数
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

    /*视频播放窗口*/
    namedWindow("Video Player", WINDOW_NORMAL);
    resizeWindow("Video Player", 800, 600); 
    fps = cap.get(CAP_PROP_FPS);
    totalFrames = cap.get(CAP_PROP_FRAME_COUNT);

    createTrackbar("Progress", "Video Player", &currentFrame, totalFrames, onTrackbarProgress);


    /*参数调节窗口*/
    namedWindow("Adjust", WINDOW_NORMAL);
    resizeWindow("Adjust", 400, 200); 

    int brightnessBar = 50;
    createTrackbar("亮度", "Adjust", &brightnessBar, 100, onTrackbarBright);
    setTrackbarPos("亮度", "Adjust", brightnessBar);

    int contrastBar = 100;
    createTrackbar("对比度", "Adjust", &contrastBar, 300, onTrackbarContrast);
    setTrackbarPos("对比度", "Adjust", contrastBar);

    int HistEqBar = 10;
    createTrackbar("直方图均衡", "Adjust", &HistEqBar, 100, onTrackbarHistEq);
    setTrackbarPos("直方图均衡", "Adjust", HistEqBar);

    int HueBar = 100;
    createTrackbar("色相", "Adjust", &HueBar, 200, onTrackbarHUE);
    setTrackbarPos("色相", "Adjust", HueBar);

    int SaturationBar = 100;
    createTrackbar("饱和度", "Adjust", &SaturationBar, 200, onTrackbarSaturation);
    setTrackbarPos("饱和度", "Adjust", SaturationBar);

    int meanBar = 1;
    createTrackbar("均值滤波核", "Adjust", &meanBar, 31, onTrackbarMeanKernelSize);
    setTrackbarPos("均值滤波核", "Adjust", meanBar);

    int medianBar = 1;
    createTrackbar("中值滤波核", "Adjust", &medianBar, 31, onTrackbarMedianKernelSize);
    setTrackbarPos("中值滤波核", "Adjust", medianBar);

    int gaussianBar = 1;
    createTrackbar("高斯滤波核", "Adjust", &gaussianBar, 31, onTrackbarGaussianKernelSize);
    setTrackbarPos("高斯滤波核", "Adjust", gaussianBar);


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

            if (key == ' ') { // 按空格键暂停或播放
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
            if (key == ' ') { // 按空格键暂停或播放
                isPlaying = !isPlaying;
            }
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
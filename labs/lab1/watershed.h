#pragma once
#include <iostream>  
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Vec3b RandomColor(int value);
Mat watershedSegmentation1(Mat frame);
Mat watershedSegmentation2(Mat frame);
Mat watershedSegmentation3(Mat frame);
Mat watershedSegmentation4(Mat frame);
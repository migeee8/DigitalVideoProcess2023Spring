﻿#include "stdio.h"
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

int main()
{
    Mat img = imread("D:\\Archive\\photos\\test.jpg");
    namedWindow("画面");
    imshow("画面", img);
    waitKey(6000);
}


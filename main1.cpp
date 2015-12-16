// est.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <opencv2\core\core_c.h>
#include <opencv2\core\core.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\imgproc\imgproc_c.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui_c.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

Mat image;
IplImage* imagetwo=0;
Mat gray;
Mat dst;
Mat tmp;

int _tmain(int argc, _TCHAR* argv[])
{
	//Canny
	image= imread("C:\\road.png", 1);
	imagetwo = cvLoadImage("C:\\road.png",CV_LOAD_IMAGE_GRAYSCALE);
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
    imshow("original", image);
	cvtColor(image, gray, CV_BGR2GRAY);
	Canny(gray, dst, 50,  250, 3);
	
	Mat InvDst=1-dst/255; //inversion
	Mat distImage=cvCreateImage(image.size(), IPL_DEPTH_32S, 1);

	distanceTransform(InvDst, distImage, CV_DIST_L2, 3);
	//normalize(distImage , distImage, 0, 1., cv::NORM_MINMAX);

	Mat *RGB_Channels = new Mat[3]; 
 	split(image, RGB_Channels); //Make R,G and B channels

	
	double MinValue=0.0;
	double MaxValue=0.0;
	minMaxLoc(distImage, &MinValue, &MaxValue);  //Location no interesting

	Mat *result = new Mat[3];
	for (int i=0; i<3; i++)
	{
		result[i].create(image.size(), CV_8U);
	}

	int MaxBorderStep = image.rows; 
	for (int k=0; k<3; k++)
	{
		Mat cur; // not changed border elements
		copyMakeBorder(RGB_Channels[k],  cur, MaxBorderStep, MaxBorderStep, MaxBorderStep, MaxBorderStep, BORDER_REPLICATE, Scalar::all(0));
		RGB_Channels[k]=cur;
		Mat prim;
		integral(RGB_Channels[k], prim);
		for (int i=0; i<image.rows; i++)
		{
			for (int j=0; j<image.cols; j++)
			{
				int MaskSize = (int)(distImage.at<float>(i,j)*0.7);
				if (MaskSize>1)
				{
					int step = MaskSize/2;
					if (MaskSize % 2 == 0) MaskSize=MaskSize+1;
					int value = prim.at<int>(MaxBorderStep + i - step,MaxBorderStep + j - step) - prim.at<int>(MaxBorderStep + i - step, MaxBorderStep + j + step + 1)
							 - prim.at<int>(MaxBorderStep + i + step + 1,MaxBorderStep + j - step) + prim.at<int>(MaxBorderStep + i + step + 1,MaxBorderStep + j + step + 1);
					result[k].at<uchar>(i, j) = (uchar)(value/(MaskSize*MaskSize));
				}
				else
				{
					result[k].at<uchar>(i, j) = RGB_Channels[k].at<uchar>(i + MaxBorderStep, j + MaxBorderStep);
				}
			}
		}
		
	}

	Mat final;
	merge(result, 3, final);
	imshow("Final", final);

	cvWaitKey(0);
	cvDestroyAllWindows();

	return 0;
}


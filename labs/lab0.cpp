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
using namespace cv;
using namespace std;

IplImage* image = 0;
IplImage* gray = 0;
IplImage* dst1 = 0;
IplImage* tmp = 0;
IplImage* image1 = 0;


int _tmain(int argc, _TCHAR* argv[])
{
	//Canny
	image= cvLoadImage("C:\\123.jpg",1);
	gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	dst1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("cvCanny", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	cvCvtColor(image, gray, CV_RGB2GRAY);
	cvCanny(gray, dst1, 100,  200, 3);
	cvShowImage("original", image);
	cvShowImage("cvCanny", dst1);
	cvShowImage("gray", gray);
	
	//BLUR
	tmp=cvCloneImage(image); // клонирование в tmp
	cvNamedWindow("Blur", CV_WINDOW_AUTOSIZE);
	cvSmooth(image, tmp, CV_GAUSSIAN, 5, 5);
	cvShowImage("Blur", tmp);

	//EqualizeHist, gray and  image - OK
	image1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvEqualizeHist(gray, image1);
	cvNamedWindow("EqualizeHist", CV_WINDOW_AUTOSIZE);
	cvShowImage("EqualizeHist", image1);


	//watershed
	//Грузим картинку
    Mat image2 = imread("C:\\123.jpg", CV_LOAD_IMAGE_COLOR);
	//выделяем грани
    Mat imageGray, imageBin;
	//В оттенки серого
    cvtColor(image, imageGray, CV_BGR2GRAY);
    threshold(imageGray, imageBin, 50, 255, THRESH_BINARY); //порог
    std::vector<std::vector<Point> > contours; //вектор контуров
    std::vector<Vec4i> hierarchy;
    findContours(imageBin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); //поиск векторов
    Mat markers(image2.size(), CV_32SC1); // создание изображения, портотипа для входного\выходного изображения
    markers = Scalar::all(0);
    int compCount = 0;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++)
    {
         drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX); // создание "карты маркеров"
    }
    std::vector<Vec3b> colorTab(compCount);
    for(int i = 0; i < compCount; i++)
    {
         colorTab[i] = Vec3b(rand()&255, rand()&255, rand()&255); // подготовка к расскраске облостей
    }
    watershed(image, markers); 
    Mat wshed(markers.size(), CV_8UC3);
    for(int i = 0; i < markers.rows; i++)
    {
         for(int j = 0; j < markers.cols; j++)
         {
            int index = markers.at<int>(i, j);
            if(index == -1)  wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else if (index == 0) wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
	        else  wshed.at<Vec3b>(i, j) = colorTab[index - 1];
          }
     }
     imshow("watershed transform", wshed);

	cvWaitKey(0);
	cvReleaseImage(&image);
	cvReleaseImage(&gray);
	cvReleaseImage(&dst1);
	cvReleaseImage(&tmp);
	cvReleaseImage(&image1);
	cvDestroyAllWindows();

	return 0;
}


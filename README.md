# weirdcard3

「レポート」

ソースコード:


IplImage* createVideoAverageImage(CvCapture* capture) {
    CvSize frameSize;
    int depth = 0;
    int channels = 0;
    CV_REDUCE_SUM;
    CV_REDUCE_AVG;
    CV_REDUCE_MAX;
    CV_REDUCE_MIN;

 
    frameSize.width = (int)cvGetCaptureProperty(
        capture, CV_CAP_PROP_FRAME_WIDTH);
    frameSize.height = (int)cvGetCaptureProperty(
        capture, CV_CAP_PROP_FRAME_HEIGHT);
    int frame_count = (int)cvGetCaptureProperty(
        capture, CV_CAP_PROP_FRAME_COUNT);
     
    ImageDataPtr sumPtr;
    ImageDataPtr imgPtr;
    IplImage* sum = 0;
 
    IplImage* captured_frame = cvQueryFrame(capture);
    if(captured_frame != 0) {
        depth = captured_frame->depth;
        channels = captured_frame->nChannels;
        sum = cvCreateImage(
            frameSize, IPL_DEPTH_32F, channels);
        sumPtr = sum;
    }
 
    while(captured_frame != 0) {
        imgPtr = captured_frame;
        for(int y = 0; y < frameSize.height; y++) {
            sumPtr.setLine(y);
            imgPtr.setLine(y);
            for(int x = 0; x < frameSize.width; x++) {
                for(int c = 0; c < channels; c++) {
                    sumPtr = sumPtr + imgPtr;
                    sumPtr++;
                    imgPtr++;
                }
            }
        }
        captured_frame = cvQueryFrame(capture);
    }
 
    IplImage* avg = cvCreateImage(frameSize, depth, channels);
    sumPtr = sum;
    imgPtr = avg;
    for(int y = 0; y < frameSize.height; y++) {
        sumPtr.setLine(y);
        imgPtr.setLine(y);
        for(int x = 0; x < frameSize.width; x++) {
            for(int c = 0; c < channels; c++) {
                imgPtr = sumPtr / frame_count;
                sumPtr++;
                imgPtr++;
            }
        }
    }
    cvReleaseImage(&sum);
    return avg;
}

from __future__ import unicode_literals, print_function

import numpy as np

void shiftDft(Mat &src, Mat &dst)
{
    Mat tmp;
    
    int cx = src.cols/2;
    int cy = src.rows/2;
    
    for(int i=0; i<=cx; i+=cx) {
        Mat qs(src, Rect(i^cx,0,cx,cy));
        Mat qd(dst, Rect(i,cy,cx,cy));
        qs.copyTo(tmp);
        qd.copyTo(qs);
        tmp.copyTo(qd);
    }
}

Mat encodeImage(Mat &src){
    Mat Re_img, Im_img, Complex_img, dft_src, dft_dst, dft_dst_p, mag_img;
    vector<Mat> mv;
    
    Size s_size = src.size();
    Re_img = Mat(s_size, CV_64F);
    Im_img = Mat::zeros(s_size, CV_64F);
    Complex_img = Mat(s_size, CV_64FC2);
    
    src.convertTo(Re_img, CV_64F);
    mv.push_back(Re_img);
    mv.push_back(Im_img);
    merge(mv, Complex_img);
    
    int dft_rows = getOptimalDFTSize(src.rows);
    int dft_cols = getOptimalDFTSize(src.cols);
    
    dft_src = Mat::zeros(dft_rows, dft_cols, CV_64FC2);
    
    Mat roi(dft_src, Rect(0, 0, src.cols, src.rows));
    Complex_img.copyTo(roi);
    
    dft(dft_src, dft_dst);
    shiftDft(dft_dst, dft_dst);
    return dft_dst;
}

Mat decodeImage(const Mat dft_dst){
    Mat dft_src, idft_img;
    vector<Mat> mv;
    double min, max;
    
    Mat dft_dst_clone=dft_dst.clone();
    shiftDft(dft_dst_clone, dft_dst_clone);
    
    idft(dft_dst_clone, dft_src);
    split(dft_src, mv);
    minMaxLoc(mv[0], &min, &max);
    idft_img = Mat(mv[0]*1.0/max);
    
    return idft_img;
}

Mat genMagImage(const Mat dft_dst){
    Mat mag_img;
    vector<Mat> mv;
    split(dft_dst, mv);
    
    magnitude(mv[0], mv[1], mag_img);
    log(mag_img+1, mag_img);
    normalize(mag_img, mag_img, 0, 1, CV_MINMAX);
    return mag_img;
}


#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace cv;
int main(int argc, char* argv[])
{
    if(argc < 2) {
        exit(-1);
    }
    cvInitSystem(argc, argv);
    CvCapture* capture = cvCreateFileCapture(argv[1]);
     
    IplImage* image_average = createVideoAverageImage(capture);
     
    cvNamedWindow("Average Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Average Image", image_average);
    cvWaitKey(0);
    Mat img1 = Mat::zeros(500, 500, CV_8UC1);

    Mat dft_dst_mask;
    dft_dst.copyTo(dft_dst_mask,mask);
    
    imshow("before",src_img);
    imshow("mag",genMagImage(dft_dst_mask));
    imshow("after",decodeImage(dft_dst_mask));
    
    waitKey(0);
    cvDestroyAllWindows();
    cvReleaseImage(&image_average);
    cvReleaseCapture(&capture);
 
    return 0;
}




参考URL
https://cvtech.cc/fourier/


説明
OpenCVで画像を読み込むところは前回から引用
途中でフーリエ変換、逆変換を行う処理を実行（void siftDft～からint main()まで）
リアルタイムに反映したものを読み込み表示する

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "face_ssd.h"
using namespace std;
using namespace cv;



int main(){

    FaceSSD face_detector;
    face_detector.init_model("./face_ssd_model_anchor.tflite");
    Mat test_img = imread("../../demo/test3.jpg");

    vector<float> score;
    vector<vector<int>> boxes;
    const clock_t begin_time = clock();

    for (int i=0; i<15; i++)
        face_detector.run_face_detection((unsigned char *) test_img.data, test_img.cols, test_img.rows, score, boxes);
    printf("time = %0.0fms\n", float( clock() - begin_time ) /  CLOCKS_PER_SEC / 15 * 1000);

    for (int i=0; i<boxes.size(); i++)
	{
		rectangle(test_img, Point(boxes[i][0], boxes[i][1]),Point(boxes[i][2], boxes[i][3]),Scalar(0,255,255),2);
        rectangle(test_img, Point(boxes[i][0]-1, boxes[i][1]-10),Point(boxes[i][0]+30, boxes[i][1]),Scalar(0,255,255),-1);
        char score_txt[32];
        sprintf(score_txt, "%0.0f%%", score[i]*100);
        putText(test_img, score_txt, Point(boxes[i][0], boxes[i][1]), 1, 0.8, Scalar(255, 0, 255), 1.5);
	}

	imshow("result1",test_img);
	imwrite("result.jpg",test_img);
    waitKey();

    return 0;
}

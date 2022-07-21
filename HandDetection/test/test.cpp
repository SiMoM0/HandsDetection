#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../include/detection.hpp"

using namespace cv;
using namespace std;

int main() {
	//DETECTOR CLASS TEST

	Detector hd ("./Dataset/rgb/");
	Prediction pred = hd.detect();
	//show all output images
	pred.show_results();
	
	// OLD TEST
	//load network
    //string format: [Windows -> "./Model/best.onnx"], [Linux -> "../Model/best.onnx"]
    /*string net_path = "./Model/best.onnx";
	auto model = dnn::readNet(net_path);
	
	int num = 10;
	for(int i=10; i<31; ++i) {
		string path = "./Dataset/rgb/" + to_string(i);
		path += ".jpg";

		Mat img = imread(path);

		vector<Mat> outputs = predict(img, model);
		printf("DONE DETECTION\n");
		vector<Rect> boxes = get_boxes(img, outputs);
		printf("BOUNDING BOXES\n");

		draw_boxes(img, boxes);

		imshow("BBOX", img);
		waitKey(0);
	}*/

	//input image
	//string path = "./Dataset/rgb/01.jpg";
	//Mat img = imread(path);
	//imshow("Input image", img);
	//waitKey(0);
//
	//vector<Mat> outputs = predict(img, model);
	//printf("DONE DETECTION\n");
	//vector<Rect> boxes = get_boxes(img, outputs);
	//printf("OBTAINED BOUNDING BOXES\n");
//
	//draw_boxes(img, boxes);
//
	//imshow("BBOX", img);
	//waitKey(0);
	
    return 0;
}

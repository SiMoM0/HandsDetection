#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../include/detection.hpp"

using namespace cv;
using namespace std;

int main() {	
	//load network
    //string format: [Windows -> "./Model/best.onnx"], [Linux -> "../Model/best.onnx"]
    string net_path = "./Model/best.onnx";
	auto model = dnn::readNet(net_path);
	
	//input image
	string path = "./Dataset/rgb/01.jpg";
	Mat img = imread(path);
	imshow("Input image", img);
	waitKey(0);

	vector<Mat> outputs = detect(img, model);
	printf("DETECTED\n");
	vector<Rect> boxes = get_boxes(img, outputs);
	printf("BOUNDING BOXES\n");

	draw_boxes(img, boxes);

	imshow("BBOX", img);
	waitKey(0);
	
    return 0;
}

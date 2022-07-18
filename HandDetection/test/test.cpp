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

	for(int i=0; i<boxes.size(); ++i) {
		Point p1 = Point(boxes[i].x, boxes[i].y);
		Point p2 = Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);
		rectangle(img, p1, p2, Scalar(0, 255, 0), 3);
	}

	imshow("BBOX", img);
	waitKey(0);
	
    return 0;
}

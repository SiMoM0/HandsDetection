#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../include/utils.hpp"
#include "../include/detection.hpp"
#include "../include/segmentation.hpp"

using namespace cv;
using namespace std;

int main() {
	//DETECTOR CLASS TEST

	//load images
	//vector<Mat> images = load_images("./Dataset/rgb");
	vector<Mat> images = load_images(DATASET_PATH);

	Detector hd;
	vector<Prediction> pred = hd.detect(images, true);

	//load ground truth boudning box
	vector<cv::String> fn;
	//glob("./Dataset/det", fn, true);
	glob(BBOX_PATH , fn, true);
	
	//test save_bbox function
	//printf("Saving bounding box test\n");
	//save_bbox(pred[0].get_bbox(), "bounding_box.txt");

	//show all output images
	/*
	for(int i=0; i<pred.size(); ++i) {
		printf("IMAGE %d\n", i+1);
		//load real boudning box
		vector<Rect> ground_truth = load_bbox(fn[i]);
		float iou = pred[i].evaluate(ground_truth);
		printf("Average IoU value: %f\n\n", iou);
		vector<Rect> bbox = pred[i].get_bbox();
		//show image with predicted boudning box
		pred[i].show_results();
	}*/
  
  //SEGMENTATION CLASS TEST
	glob(MASK_PATH, fn, true);
  
	for (int i = 0; i < pred.size(); i++) {
		Segmenter p(pred[i], fn[i]);
		p.segment_regions();
		p.write_segmented();
		p.pixel_accuracy();
	}
	



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

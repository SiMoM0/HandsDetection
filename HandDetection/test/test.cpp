#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../include/utils.hpp"
#include "../include/detection.hpp"
#include "../include/segmentation.hpp"

int main() {
	//DETECTOR CLASS TEST

	//load images
	//vector<Mat> images = load_images("./Dataset/rgb");
	std::vector<cv::Mat> images = load_images(DATASET_PATH);

	Detector hd;
	std::vector<Prediction> pred = hd.detect(images, true);

	//load ground truth boudning box
	std::vector<cv::String> fn;
	//glob("./Dataset/det", fn, true);
	cv::glob(BBOX_PATH , fn, true);
	
	//test save_bbox function
	//printf("Saving bounding box test\n");
	//save_bbox(pred[0].get_bbox(), "bounding_box.txt");

	//show all output images
	
	for(int i=0; i<pred.size(); ++i) {
		std::printf("IMAGE %d\n", i+1);
		//load real boudning box
		std::vector<cv::Rect> ground_truth = load_bbox(fn[i]);
		float iou = pred[i].evaluate(ground_truth);
		printf("Average IoU value: %f\n\n", iou);
		std::vector<cv::Rect> bbox = pred[i].get_bbox();
		//show image with predicted boudning box
		pred[i].show_results();

		// save .txt bounding boxes
		std::string filenametxt = RESULTSTXT + std::to_string(i + 1) + ".txt";
		save_bbox(bbox, filenametxt);
		// save detected image
		std::string filenamedet = RESULTSDET + std::to_string(i + 1) + ".jpg";
		cv::imwrite(filenamedet, pred[i].get_output());
	}
  
  //SEGMENTATION CLASS TEST
	cv::glob(MASK_PATH, fn, true);

  
	for (int i = 0; i < pred.size(); i++) {
		Segmenter seg(pred[i], fn[i]);
		// segment the region of the image
		seg.segment_regions();
		// write the segmented image in the output_variable of the class
		seg.write_segmented();
		// print the results of the segmentation
		seg.print_results(i);

		// save segmened image
		std::string filenameseg = RESULTSSEG+ std::to_string(i + 1) + ".jpg";
		cv::imwrite(filenameseg, seg.get_output());
	}

    return 0;
}

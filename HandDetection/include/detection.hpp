//Header file for the detection task
#ifndef DETECTION
#define DETECTION

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

/**
 * Perform detection on an input image and returns the output of the network
 * @param img input image on which make the detection
 * @param network the network model use for the detection
 * @return a vector of the output detection as a cv::Mat object
 */
std::vector<cv::Mat> detect(const cv::Mat& img, cv::dnn::Net& network);

/**
 * Obtain the bounding box from the output of the network, in the following format:
 * [x, y, width, height]
 * @param img input image
 * @param detections output vector of the network
 * @return vector of all bounding box
 */
std::vector<cv::Rect> get_boxes(const cv::Mat& img, const std::vector<cv::Mat>& detections);

/**
 * Draw bounding box on an image
 * @param img input image
 * @param boxes vector of bounding box (cv::Rect) to be drawn on the image
 */
void draw_boxes(cv::Mat& img, const std::vector<cv::Rect>& boxes);

#endif //DETECTION
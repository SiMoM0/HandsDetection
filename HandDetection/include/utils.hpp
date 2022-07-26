/**
 * @file utils.hpp
 * 
 * @brief Header file for useful variables and various functions
 * 
 * @author //TODO: ADD AUTHOR
 */
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>

//Network model file path
const std::string NETWORK_PATH = MODEL_PATH;

//Image width
const float WIDTH = 640.0;
//Image height
const float HEIGHT = 640.0;
//Dimension of network output
const int DIMENSION = 25200;

//Confidence threshold for the detection
const float CONFIDENCE_THRESHOLD = 0.45;
//Score threshold for the detection of a hand
const float SCORE_THRESHOLD = 0.45;
//Non maximum suppresion threshold for bounding box
const float NMS_THRESHOLD = 0.45;

//Colors
const cv::Scalar BLACK (0, 0, 0);
const cv::Scalar WHITE (255, 255, 255);
const cv::Scalar RED (0, 0, 255);
const cv::Scalar BLUE (255, 0, 0);
const cv::Scalar GREEN (0, 255, 0);
const cv::Scalar YELLOW (0, 255, 255);
const cv::Scalar CYAN (255, 255, 0);
const cv::Scalar MAGENTA (255, 0, 255);
//vector of colors
const std::vector<cv::Scalar> COLORS {RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA};

/**
 * Show a single image
 * @param img input image to display
 * @param window_name name of the window, defualt = "Image"
 */
void show_image(const cv::Mat& img, const std::string& window_name = "Image");

/**
 * Show a set of images
 * @param images vector if images of type cv::Mat
 */
void show_images(const std::vector<cv::Mat>& images, const std::string& window_name);

/**
 * Load a single image from its path
 * @param path path of the image to be loaded
 * @return image as cv::Mat object
 */
cv::Mat load_image(const std::string& path);

/**
 * Upload a set of images from a given input directory path
 * @param path the directory path where the images are stored
 * @return vector of images as cv::Mat object
 */
std::vector<cv::Mat> load_images(const std::string& path);

/**
 * Load bounding box from a txt file
 * @param file_path path of the txt file
 * @return list of bounding box as std::vector<cv::Rect> type
 */
std::vector<cv::Rect> load_bbox(const std::string& file_path);

/**
 * Evaluate IoU metric given the ground truth and predicted bounding box
 * @param prediction the predicted bounding box
 * @param ground_truth the real bounding box
 * @return IoU value as float value
 */
float IoU(const cv::Rect& prediction, const cv::Rect& ground_truth);

/**
 * Evaluate IoU metric given a list of ground truth and a list of predicted bounding box
 * @param prediction the vector of predicted bounding box
 * @param ground_truth the vector of real bounding box
 * @return IoU value as float value
 */
float IoU(const std::vector<cv::Rect>& prediction, const std::vector<cv::Rect>& ground_truth);

/**
 * Save the list of bounding box in a txt file
 * @param boxes list of bounding box as std::vector<cv::Rect> object
 * @param file_name name of the file to be saved
 * @return true if the operation succeed, fals otherwise
 */
bool save_bbox(const std::vector<cv::Rect> boxes, const std::string& file_name);

#endif //UTILS_H
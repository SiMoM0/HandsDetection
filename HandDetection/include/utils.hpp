//Header file for useful variables and various functions
#ifndef UTILS
#define UTILS

#include <opencv2/core.hpp>

//Image width
const float WIDTH = 640.0;
//Image height
const float HEIGHT = 640.0;
//Dimension of network output
const int DIMENSION = 25200;

//Confidence threshold for the detection
const float CONFIDENCE_THRESHOLD = 0.4;
//Score threshold for the detection of a hand
const float SCORE_THRESHOLD = 0.4;
//Non maximum suppresion threshold for bounding box
const float NMS_THRESHOLD = 0.4;

//Colors
const cv::Scalar BLACK (0, 0, 0);
const cv::Scalar WHITE (255, 255, 255);
const cv::Scalar RED (0, 0, 255);
const cv::Scalar BLUE (255, 0, 0);
const cv::Scalar GREEN (0, 255, 0);

#endif //UTILS
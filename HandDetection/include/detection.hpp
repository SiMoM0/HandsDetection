/**
 * @file detection.hpp
 * 
 * @brief Classes and functions for the hand detection
 * 
 * @author //TODO: ADD AUTHOR
 */

#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

/**
 * Class Predictions
 * 
 * Contains all the input images, list of bounding box and labeled images
 * output from the a Detector object
 */
class Prediction {
    public:
        /**
         * Create Prediction object
         * @param input_images list of input images
         * @param bounding_box list of all bounding box for each input image
         * @param output_images list of all images with corresponding bounding box drawn
         */
        Prediction(const std::vector<cv::Mat>& input_images, const std::vector<std::vector<cv::Rect>>& bounding_box, const std::vector<cv::Mat>& output_images);

        /**
         * Number of images contained in the prediction. Same number of input and output images.
         * @return integer number correponding to the size
         */
        int size() {return input_images.size();}

        /**
         * Get the list of input images
         * @return vector of input images
         */
        std::vector<cv::Mat> get_input() {return input_images;}

        /**
         * Get the list of all bounding box
         * @return list of bounding box for each image
         */
        std::vector<std::vector<cv::Rect>> get_bbox() {return bounding_box;}

        /**
         * Get the vector of output images
         * @return output images with bounding box drawn
         */
        std::vector<cv::Mat> get_output() {return output_images;}

    private:
        std::vector<cv::Mat> input_images;
        std::vector<std::vector<cv::Rect>> bounding_box;
        std::vector<cv::Mat> output_images;
};

/**
 * Class for Hand Detection
 * 
 * Implement all the functions for performing hand detection with bounding box
 */
class Detector {
    public:
        /**
         * Create a Detector object
         */
        Detector();

        /**
         * Create a Detector object and load images stored in the input path
         * @param images_path directory path where images are stored
         */
        Detector(const std::string& images_path);

        /**
         * Load images from a given path
         * @param images_path directory path where images are stored
         */
        void add_images(const std::string& images_path);

        /**
         * Load images from a vector of images
         * @param images vector of images of type std::vector<cv::Mat>
         */
        void add_images(const std::vector<cv::Mat> images);

        /**
         * Load a single image from its path
         * @param image_path path of the given image
         */
        void add_image(const std::string& image_path);

        /**
         * Load a single image
         * @param image a cv::Mat object to load
         */
        void add_image(const cv::Mat& image);

        /**
         * Perform the detection on all the images loaded
         * @return a Prediction object type
         */
        Prediction detect();

        //TODO: add destructor, similar and other functions

    private:
        cv::dnn::Net network;
        std::vector<cv::Mat> input_images;
        std::vector<std::vector<cv::Mat>> net_outputs;
        std::vector<std::vector<cv::Rect>> bounding_box;
        std::vector<cv::Mat> output_images;

        /**
         * Perform forward pass on a single image and returns the output of the network
         * @param img input image to be feed to the network
         * @return output of the network as std::vector<cv::Mat>
         */
        std::vector<cv::Mat> forward_pass(const cv::Mat& img);

        /**
         * Convert a single output of the network in bounding box format and save them
         * [x, y, width, height]
         * @param img input image
         * @param detections output of the network
         * @return bounding box as std::vector<cv::Rect> object
         */
        std::vector<cv::Rect> convert_boxes(const cv::Mat& img, const std::vector<cv::Mat>& detections);

        /**
         * Draw bounding box and generate a single output image
         * @param img input image
         * @param boxes list of bounding box
         * @return output image with bounding box drawn
         */
        cv::Mat generate_labels(const cv::Mat& img, const std::vector<cv::Rect>& boxes);
};

/* OLD FUNCTIONS*/

/**
 * Perform forward pass on a single input image and returns the output of the network
 * @param img input image on which make the detection
 * @param network the network model use for the detection
 * @return a vector of the output detection as a cv::Mat object
 */
//std::vector<cv::Mat> predict(const cv::Mat& img, cv::dnn::Net& network);

/**
 * Obtain the list of all bounding box from the output of the network, in the following format:
 * [x, y, width, height]
 * @param img input image
 * @param detections output vector of the network
 * @return vector of all bounding box
 */
//std::vector<cv::Rect> get_boxes(const cv::Mat& img, const std::vector<cv::Mat>& detections);

/**
 * Draw bounding box on an image
 * @param img input image
 * @param boxes vector of bounding box (cv::Rect) to be drawn on the image
 */
//void draw_boxes(cv::Mat& img, const std::vector<cv::Rect>& boxes);

/**
 * General detection function that predict bounding boxes on input image and output them drawn in the output image
 * @param input_image input image on which perform the detection
 * @param output_image image with the bounding box drawn
 * @param network network that perform the detection
 */
//void detect(const cv::Mat& input_image, cv::Mat& output_image, cv::dnn::Net& network);

//TODO: functions for list of images
//TODO: IoU function

#endif //DETECTION_H
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
 * Contains the input image, list of bounding box and labeled image
 * output from the a Detector object
 */
class Prediction {
    public:
        /**
         * Create Prediction object
         * @param input_image input image
         * @param bounding_box list of all bounding box fo input image
         * @param output_image image with corresponding bounding box drawn
         */
        Prediction(const cv::Mat& input_image, const std::vector<cv::Rect>& bounding_box, const cv::Mat& output_image);

        /**
         * Destructor for Prediction class
         */
        ~Prediction() {}

        /**
         * Get the input image
         * @return cv::Mat input image
         */
        cv::Mat get_input() {return input_image;}

        /**
         * Get the list of all bounding box
         * @return list of bounding box for input image
         */
        std::vector<cv::Rect> get_bbox() {return bounding_box;}

        /**
         * Get the output image with bouding box
         * @return cv::Mat output image with bounding box drawn
         */
        std::vector<cv::Mat> get_output() {return output_image;}

        /**
         * Display input image
         */
        void show_input();

        /**
         * Display image with corresponding bounding box
         */
        void show_results();

        /**
         * Display image with predicted bounding box and real boudning box
         * @param ground_truth real boudning box
         */
        void show_results(const std::vector<cv::Rect>& ground_truth);

        /**
         * Evaluate predicted bounding box compared to the ground truth using IoU metric
         * @param ground_truth true bounding box of image
         * @return IoU float value
         */
        float evaluate(const std::vector<cv::Rect>& ground_truth);

    private:
        cv::Mat input_image;
        std::vector<cv::Rect> bounding_box;
        cv::Mat output_image;
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
         * Destructor for Detector class
         */
        ~Detector() {}

        /**
         * Number of images contained
         * @return integer number corresponding to the number of input images
         */
        int size() {return input_images.size();}

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
         * @return a vector of Prediction object type
         */
        std::vector<Prediction> detect();

        /**
         * Perform bounding box detection on a single image
         * @param img input image
         * @return Prediction object
         */
        Prediction detect(const cv::Mat& img);

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
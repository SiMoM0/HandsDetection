/**
 * @file detection.hpp
 * 
 * @brief Classes and functions for the hand detection
 * 
 * @author //TODO: Grotto Gionata, Mosco Simone, Pisacreta Giulia
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
         * Indicates if the prediction contain some hands
         * @return true if the input image contains hands, false otherwise
         */
        bool contains_hands();

        /**
         * Returns the number of hands found in the input image
         * @return integer value representing number of hands
         */
        int hands_number();

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
         * Create Detector object with custom network
         * @param net custom network for Detector object
         */
        Detector(const cv::dnn::Net& net);

        /**
         * Destructor for Detector class
         */
        ~Detector() {}

        /**
         * Perform bounding box detection on a single image
         * @param img input image
         * @param verbose print information, default = false
         * @return Prediction object
         */
        Prediction detect(const cv::Mat& img, const bool& verbose = false);

        /**
         * Perform the detection on all the input images
         * @param images vector of input images
         * @param verbose print information, default = false
         * @return a vector of Prediction object type
         */
        std::vector<Prediction> detect(const std::vector<cv::Mat>& images, const bool& verbose = false);

        /**
         * Load a single or a set of images from path and perform detection
         * @param path path of image or directory of images
         * @param dir true if specified directory, false if single image to be load
         * @return vector of prediction for both option
         */
        std::vector<Prediction> detect(const std::string& path, const bool& dir, const bool& verbose = false);

    private:
        cv::dnn::Net network;

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

#endif //DETECTION_H
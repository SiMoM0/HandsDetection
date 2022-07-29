/**
 * @file segmentation.hpp
 * 
 * @brief Classes and functions for the hand segmentation
 * 
 * @author //TODO: Grotto Gionata, Mosco Simone, Pisacreta Giulia
 */
#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "../include/detection.hpp"

//Colors
const cv::Vec3b BLACK_ (0, 0, 0);
const cv::Vec3b WHITE_ (255, 255, 255);
const cv::Vec3b RED_ (0, 0, 255);
const cv::Vec3b BLUE_ (255, 0, 0);
const cv::Vec3b GREEN_ (0, 255, 0);
const cv::Vec3b YELLOW_ (0, 255, 255);
const cv::Vec3b CYAN_ (255, 255, 0);
const cv::Vec3b MAGENTA_ (255, 0, 255);
//vector of colors to color the different hands
const std::vector<cv::Vec3b> COL {RED_, GREEN_, BLUE_, YELLOW_, CYAN_, MAGENTA_};

class Segmenter {
    public:

        /**
         * Create Segmenter object
         * @param predicted_image Predicted object used to load the image and the bouding boxes
         * @param path_mask path of true masked image
         */
        Segmenter(Prediction& predicted_image, std::string path_mask);

        /**
         * Destructor for Segmenter class
         */
        ~Segmenter() {}

        /**
         * Apply the segmentation method on the regions and save them in a vector
         */
        void segment_regions();
        
        /**
         * Rewrite the segmented image in the output image and print it
         */
        void write_segmented();

        /**
         * Return rate of segmentation
         */
        float pixel_accuracy();

        /**
         * Print the pixel accuracy of the segmentation and the final segmented image
         * @param counter inxed of the image segmented
         */
        void print_results(const int& counter);

        /**
         * Get the output image with segmented regions
         * @return cv::Mat output image with colored regions
         */
        cv::Mat get_output() {return output_image;}






    
    private:
        cv::Mat masked_image;
        cv::Mat true_mask;
        cv::Mat output_image;
        std::vector<cv::Rect> bounding_boxes;
        std::vector<cv::Mat> hand_regions;
        std::vector<cv::Mat> hand_segmented;
        std::vector<cv::Mat> mask_regions;

        /**
         * Get all the boxes in the image that contains a hand
         */
        void get_box_region();
};

/**
 * Function that performs the segmentation using the HSV color space (Hue, 
 * Saturation, Value). Check the element in range of the Hue value.
 * 
 * @param input input image
 * @param output segmented image
 */
void hsv_segmentation(const cv::Mat& input, cv::Mat& output);

/**
 * Function that performs the segmentation using the Region Growing technique.
 * It uses the result of the multicolor_segmentation to find the mask to erode and
 * start the algorithm
 * 
 * @param input input image
 * @param mask mask region
 * @param similarity distance criteria between neighbour points, uchar value
 */
void region_growing(const cv::Mat& input, cv::Mat& mask, uchar similarity);

/**
 * Function that performs the segmentation using the YCbCr color space
 * 
 * @param input input image
 * @param output segmented image
 */
void ycbcr_segmentation(const cv::Mat& input, cv::Mat& output);

/**
 * Function that performs the segmentation using the BGR color space
 * 
 * @param input input image
 * @param output segmented image
 */
void bgr_segmentation(const cv::Mat& input, cv::Mat& output);

/**
 * Function that performs the segmentation using the Otsu method 
 * 
 * @param input input image
 * @param output output image
 * @param ksize blur size
 */
void otsu_segmentation(const cv::Mat& input, cv::Mat& output, const int ksize);

/**
 * Function that performs the segmentation using a clustering technique: K-means
 * 
 * @param input input image
 * @param output segmented image
 * @param k number of clusters
 * @param color set true if input image is colorful
 */
void kmeans_segmentation(const cv::Mat& input, cv::Mat& output, const int k, const bool color=false);

/**
 * Function that performs the segmentation using the HSL (Hue, Saturation, Lightness) color space
 * 
 * @param input input image
 * @param output segmented image
 */
void hsl_segmentation(const cv::Mat& input, cv::Mat& output);

/**
 * Function that finds edges in the image 
 * 
 * @param input input image
 * @param output output image
 * @param sz blur size
 * @param kernel_size canny kernel size
 */
void canny_edge(const cv::Mat& input, cv::Mat& output, int sz, int kernel_size);

void watershed_segmentation(const cv::Mat& input, cv::Mat& output);

/**
 * Use RGB, HSV and YCbCr color spaces threshold to find the hand color and
 * intensity in the image
 * 
 * @param input input image
 * @param output segmented image
 */
void multicolor_segmentation(const cv::Mat& input, cv::Mat& output);


/**
 * Blur only the element that are inside the white region in the mask
 * 
 * @param input input image
 * @param mask mask of the input image
 * @param output resulted blurred image
 */
void blur_mask(const cv::Mat& input, cv::Mat& mask, cv::Mat& output);

/**
 * Return the normalized number of white pixel in the mask
 * 
 * @param mask input mask
 * @return float normalized value of white pixels
 */
float check_mask(const cv::Mat& mask);

#endif //SEGMENTATION_H
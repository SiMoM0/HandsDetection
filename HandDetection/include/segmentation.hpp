//Header file for the segmentation task
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
//vector of colors
const std::vector<cv::Vec3b> COL {RED_, GREEN_, BLUE_, YELLOW_, CYAN_, MAGENTA_};

class Segmenter {
    public:

        /**
         * Create Segmenter object
         * @param predicted_image Predicted object used to load the image and the bouding boxes
         */
        Segmenter(Prediction& predicted_image);

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






    
    private:
        cv::Mat masked_image;
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

void otsuSegmentation(const cv::Mat& input, cv::Mat& output, const int ksize);

void kmeansSegmentation(const cv::Mat& input, cv::Mat& output, const int k, const bool color=false);

void regionGrowing(const cv::Mat& input, cv::Mat& mask, const int ksize, uchar similarity);

void hsvSegmentation(const cv::Mat& input, cv::Mat& output);

void Dilation(const cv::Mat& src, const cv::Mat& dst, int dilation_elem, int dilation_size);

void ycbSegmentation(const cv::Mat& input, cv::Mat& output);

void bgrSegmentation(const cv::Mat& input, cv::Mat& output);

void hslSegmentation(const cv::Mat& input, cv::Mat& output);

void htsSegmentation(const cv::Mat& input, cv::Mat& output);

void cannyEdge(const cv::Mat& input, cv::Mat& output, int sz, int kernel_size);

void watershedSegmentation(const cv::Mat& input, cv::Mat& output);

void multicolorSegmentation(const cv::Mat& input, cv::Mat& output);

#endif //SEGMENTATION_H
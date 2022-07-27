//Implementation of the header "utils.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void show_image(const cv::Mat& img, const std::string& window_name = "Image") {
    cv::namedWindow(window_name);
    cv::imshow(window_name, img);
    cv::waitKey(0);
}

void show_images(const std::vector<cv::Mat>& images, const std::string& window_name = "Image") {
    //create window with input name
    cv::namedWindow(window_name);
    for(int i=0; i<images.size(); ++i) {
        cv::imshow(window_name, images[i]);
        cv::waitKey(0);
    }
}

cv::Mat load_image(const std::string& path) {
    //try to read the image
    cv::Mat img = cv::imread(path);
        if(img.empty())
            std::printf("Could not open images at %s", path);
            //throw exception
            throw std::invalid_argument("Problem loading image\n");
    return img;
}

std::vector<cv::Mat> load_images(const std::string& path) {
    //try to read the images
    cv::String data_path (path);
    std::vector<cv::String> fn;
    cv::glob(data_path, fn, true);

    //get all images in the directory and store in a vector
    std::vector<cv::Mat> images;
    for(int i=0; i<fn.size(); ++i) {
        cv::Mat img = cv::imread(fn[i]);
        //check loaded image
        if(img.empty())
            std::printf("Could not load image at %s\n", fn[i]);
        else
            images.push_back(img);
    }
    //return vector of loaded images
    return images;
}

std::vector<cv::Rect> load_bbox(const std::string& file_path) {
    //vector to store bounding box
    std::vector<cv::Rect> bounding_box;
    //get file
    std::ifstream infile(file_path);
    //rectangle variable
    int x, y, width, height;
    while(infile >> x >> y >> width >> height) {
        cv::Rect box (x, y, width, height);
        bounding_box.push_back(box);
    }

    return bounding_box;
}

float IoU(const cv::Rect& prediction, const cv::Rect& ground_truth) {
    //intersection between two bounding box
    cv::Rect intersection = prediction & ground_truth;
    int inter_area = intersection.area();
    //union of the two rectangle
    int union_area = prediction.area() + ground_truth.area() - intersection.area();
    //iou float value
    float iou = (float) inter_area/union_area;

    return iou;
}

float IoU(const std::vector<cv::Rect>& prediction, const std::vector<cv::Rect>& ground_truth) {
    float iou = 0;
    //check if they have the same dimension
    if(prediction.size() != ground_truth.size()) {
        std::printf("Different number of bounding box found, IoU value might be incorrect\n");
    }
    //loop through all the bounding box
    for(int i=0; i<prediction.size(); ++i) {
        //consider the max IoU for each rectangle in order to match them correctly
        float max = 0;
        for(int j=0; j<ground_truth.size(); ++j) {
            float score = IoU(prediction[i], ground_truth[j]);
            //update max
            if(score > max) {
                max = score;
            }
        }
        //add score found to iou
        iou += max;
    }
    //get the maximum number of bounding box between the true and predicted
    int box_num = std::max(prediction.size(), ground_truth.size());

    return iou/box_num;
}

void save_bbox(const std::vector<cv::Rect> boxes, const std::string& file_name) {
    //create and open txt file
    std::ofstream output_file (file_name);
    //write each line
    for(int i=0; i<boxes.size(); ++i) {
        int x = boxes[i].x;
        int y = boxes[i].y;
        int width = boxes[i].width;
        int height = boxes[i].height;
        //if last element don't use \n
        if(i != boxes.size()-1) {
            output_file << x << " " << y << " " << width << " " << height << "\n";
        } else {
            output_file << x << " " << y << " " << width << " " << height;
        }
    }
    //close file
    output_file.close();
}
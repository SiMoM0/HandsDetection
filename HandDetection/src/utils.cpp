//Implementation of the header "utils.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void show_image(const cv::Mat& img, const std::string& window_name = "Image") {
    cv::namedWindow(window_name);
    cv::imshow(window_name, img);
    cv::waitKey(0);
}

void show_images(const std::vector<cv::Mat>& images, const std::string& window_name) {
    //create window with input name
    cv::namedWindow(window_name);
    for(int i=0; i<images.size(); ++i) {
        cv::imshow(window_name, images[i]);
        cv::waitKey(0);
    }
}

cv::Mat load_image(const std::string& path) {
    //TODO: add path check
    cv::Mat img = cv::imread(path);
        if(img.empty())
            std::printf("Could not open images at %s", path);
            //FIXME: throw exception or return an empty/black image (?)
    return img;
}

std::vector<cv::Mat> load_images(const std::string& path) {
    //TODO: add path check
    cv::String data_path (path);
    std::vector<cv::String> fn;
    cv::glob(data_path, fn, true);

    //get all images in the directory and store in a vector
    std::vector<cv::Mat> images;
    for(int i=0; i<fn.size(); ++i) {
        cv::Mat img = cv::imread(fn[i]);
        //check loaded image
        if(img.empty())
            std::printf("Could not load image at %s", fn[i]);
        else
            images.push_back(img);
    }
    //return vector of loaded images
    return images;
}
//Implementation of the header "detection.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include "../include/utils.hpp"

std::vector<cv::Mat> detect(const cv::Mat& img, cv::dnn::Net& network) {
    //Convert input image to blob
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1./255., cv::Size(WIDTH, HEIGHT), cv::Scalar(), true, false);
    //Set input image to the network
    network.setInput(blob);

    //Get output detections
    std::vector<cv::Mat> outputs;
    network.forward(outputs, network.getUnconnectedOutLayersNames());

    return outputs;
}

std::vector<cv::Rect> get_boxes(const cv::Mat& img, const std::vector<cv::Mat>& detections) {
    //vector of all confidence values
    std::vector<float> confidences;
    //vector of all bounding box found
    std::vector<cv::Rect> bbox;
    
    //image infos
    int rows = img.rows;
    int cols = img.cols;

    //get detections data
    float* data = (float*) detections[0].data;

    //TODO: delete
    float max = 0.0;

    //Iterate through the all the detections
    for(int i=0; i<DIMENSION; ++i) {
        //get confidence of the current detection
        float confidence = data[4];
        if(confidence > max) max = confidence;
        if(confidence >= CONFIDENCE_THRESHOLD) {
            //get score
            float score = data[5];
            if(score >= SCORE_THRESHOLD) {
                //add confidences
                confidences.push_back(confidence);
                //get box values
                float x_center = data[0];
                float y_center = data[1];
                float w = data[2];
                float h = data[3];
                //convert to output format
                int x = int((x_center - .5*w) * cols/WIDTH);
                int y = int((y_center - .5*h) * rows/HEIGHT);
                int width = int(w*cols/WIDTH);
                int height = int(h*rows/HEIGHT);

                //add box to the vector
                bbox.push_back(cv::Rect(x, y, width, height));
            }
        }
        //next detection
        data += 6;
    }
    //TODO: delete
    printf("Max confidence: %f\n", max);

    return bbox;
}
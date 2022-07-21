//Implementation of the header "detection.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "../include/utils.hpp"
#include "../include/detection.hpp"

//Detector class and functions definition

Detector::Detector() {
    network = cv::dnn::readNet(NETWORK_PATH);
}

Detector::Detector(const std::string& images_path) {
    network = cv::dnn::readNet(NETWORK_PATH);
    input_images = load_images(images_path);
}

void Detector::add_images(const std::string& images_path) {
    std::vector<cv::Mat> images = load_images(images_path);
    for(int i=0; i<images.size(); ++i) {
        input_images.push_back(images[i]);
    }
}

void Detector::add_images(const std::vector<cv::Mat> images) {
    for(int i=0; i<images.size(); ++i) {
        input_images.push_back(images[i]);
    }
}

void Detector::add_image(const std::string& image_path) {
    input_images.push_back(load_image(image_path));
}

void Detector::add_image(const cv::Mat& image) {
    input_images.push_back(image);
}

std::vector<cv::Mat> Detector::forward_pass(const cv::Mat& img) {
    //Convert input image to blob
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1./255., cv::Size(WIDTH, HEIGHT), cv::Scalar(), true, false);
    //Set input image to the network
    network.setInput(blob);
    //Get and save output detections
    std::vector<cv::Mat> outputs;
    network.forward(outputs, network.getUnconnectedOutLayersNames());

    return outputs;
}

std::vector<cv::Rect> Detector::convert_boxes(const cv::Mat& img, const std::vector<cv::Mat>& detections) {
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
    for(int j=0; j<DIMENSION; ++j) {
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

    //apply non maximum suppresion to get only one bounding box
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bbox, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    //keep only the best bbox
    std::vector<cv::Rect> best_bbox;
    for(int i=0; i<indices.size(); ++i) {
        best_bbox.push_back(bbox[indices[i]]);
    }

    //bounding box of current image
    printf("Number of final bounding box: %d\n", best_bbox.size());
    
    return best_bbox;
}

cv::Mat Detector::generate_labels(const cv::Mat& img, const std::vector<cv::Rect>& boxes) {
    //generate a temp clone of the image
    cv::Mat temp = img.clone();
    //number of bounding box
    int box_num = boxes.size();
    for(int i=0; i<box_num; ++i) {
        //Extract and compute the attributes
        int x = boxes[i].x;
        int y = boxes[i].y;
        int width = boxes[i].width;
        int height = boxes[i].height;
        //printf("LABEL: %d %d %d %d\n", x, y, width, height);
        
        //Create the two points for the rectangle
		cv::Point p1 (x, y);
		cv::Point p2 (x + width, y + height);
		cv::rectangle(temp, p1, p2, GREEN, 3);
	}
    return temp;
}

Prediction Detector::detect() {
    for(int i=0; i<input_images.size(); ++i) {
        cv::Mat img = input_images[i];
        //call the three functions for the general prediction
        std::vector<cv::Mat> output = forward_pass(img);
        //Store output of the network
        net_outputs.push_back(output);
        std::vector<cv::Rect> bbox = convert_boxes(img, output);
        //Store bounding box
        bounding_box.push_back(bbox);
        cv::Mat det_img = generate_labels(img, bbox);
        //Store labeled image
        output_images.push_back(det_img);
    }

    return Prediction(input_images, bounding_box, output_images);
}

cv::Mat Detector::detect(const cv::Mat& img) {
    std::vector<cv::Mat> output = forward_pass(img);
    std::vector<cv::Rect> bbox = convert_boxes(img, output);
    return generate_labels(img, bbox);
}

//Prediction class and functions definition

Prediction::Prediction(const std::vector<cv::Mat>& input_images, const std::vector<std::vector<cv::Rect>>& bounding_box, const std::vector<cv::Mat>& output_images) {
    this->input_images = input_images;
    this->bounding_box = bounding_box;
    this->output_images = output_images;
}

void Prediction::show_inputs() {
    show_images(input_images, "Input Image");
}

void Prediction::show_results() {
    show_images(output_images, "Bounding Box");
}

/* OLD IMPLEMENTATION
std::vector<cv::Mat> predict(const cv::Mat& img, cv::dnn::Net& network) {
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

    //apply non maximum suppresion to get only one bounding box
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bbox, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    //keep only the best bbox
    std::vector<cv::Rect> best_bbox;
    for(int i=0; i<indices.size(); ++i) {
        best_bbox.push_back(bbox[indices[i]]);
    }

    printf("Number of final bounding box: %d\n", best_bbox.size());

    return best_bbox;
}

void draw_boxes(cv::Mat& img, const std::vector<cv::Rect>& boxes) {
    //number of bounding box
    int box_num = boxes.size();
    for(int i=0; i<box_num; ++i) {
        //Extract and compute the attributes
        int x = boxes[i].x;
        int y = boxes[i].y;
        int width = boxes[i].width;
        int height = boxes[i].height;
        printf("LABEL: %d %d %d %d\n", x, y, width, height);
        //Create the two points for the rectangle
		cv::Point p1 (x, y);
		cv::Point p2 (x + width, y + height);
		cv::rectangle(img, p1, p2, GREEN, 3);
	}
}*/
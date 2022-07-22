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
		cv::rectangle(temp, p1, p2, RED, 3);
	}
    return temp;
}

std::vector<Prediction> Detector::detect() {
    //output vector of Prediction objects
    std::vector<Prediction> pred;
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

        //store new prediction
        pred.push_back(Prediction(input_images[i], bbox, det_img));
    }

    return pred;
}

Prediction Detector::detect(const cv::Mat& img) {
    std::vector<cv::Mat> output = forward_pass(img);
    std::vector<cv::Rect> bbox = convert_boxes(img, output);
    cv::Mat det_img = generate_labels(img, bbox);

    return Prediction(img, bbox, det_img);
}

//Prediction class and functions definition

Prediction::Prediction(const cv::Mat& input_image, const std::vector<cv::Rect>& bounding_box, const cv::Mat& output_image) {
    this->input_image = input_image;
    this->bounding_box = bounding_box;
    this->output_image = output_image;
}

void Prediction::show_input() {
    show_image(input_image, "Input Image");
}

void Prediction::show_results() {
    show_image(output_image, "Bounding Box");
}

void Prediction::show_results(const std::vector<cv::Rect>& ground_truth) {
    //temp image
    cv::Mat temp = output_image.clone();
    for(int i=0; i<ground_truth.size(); ++i) {
        //Extract and compute the attributes
        int x = ground_truth[i].x;
        int y = ground_truth[i].y;
        int width = ground_truth[i].width;
        int height = ground_truth[i].height;
        //printf("LABEL: %d %d %d %d\n", x, y, width, height);
        
        //Create the two points for the rectangle
		cv::Point p1 (x, y);
		cv::Point p2 (x + width, y + height);
		cv::rectangle(temp, p1, p2, GREEN, 3);
    }
    //show image
    show_image(temp, "Detection");
}

float Prediction::evaluate(const std::vector<cv::Rect>& ground_truth) {
    return IoU(bounding_box, ground_truth);
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
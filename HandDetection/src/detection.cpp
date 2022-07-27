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

Detector::Detector(const cv::dnn::Net& net) {
    network = net;
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
    //TODO: comment
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
    //TODO: comment
    //printf("Max confidence: %f\n", max);

    //apply non maximum suppresion to get only one bounding box
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bbox, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    //keep only the best bbox
    std::vector<cv::Rect> best_bbox;
    for(int i=0; i<indices.size(); ++i) {
        best_bbox.push_back(bbox[indices[i]]);
    }

    //bounding box of current image
    //printf("Number of final bounding box: %d\n", best_bbox.size());
    
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
		cv::rectangle(temp, p1, p2, COLORS[i], 2);
	}
    return temp;
}

Prediction Detector::detect(const cv::Mat& img, const bool& verbose) {
    if(verbose) {
        std::printf("PERFORMING DETECTION ON IMAGE\n");
    }
    std::vector<cv::Mat> output = forward_pass(img);
    std::vector<cv::Rect> bbox = convert_boxes(img, output);
    cv::Mat det_img = generate_labels(img, bbox);

    return Prediction(img, bbox, det_img);
}

std::vector<Prediction> Detector::detect(const std::vector<cv::Mat>& images, const bool& verbose) {
    if(verbose) {
        std::printf("PERFORMING DETECTION ON IMAGES\n");
    }
    //output vector of predictions
    std::vector<Prediction> output;
    for(int i=0; i<images.size(); ++i) {
        //call single detect function
        Prediction pred = detect(images[i]);
        output.push_back(pred);
    }

    return output;
}

std::vector<Prediction> Detector::detect(const std::string& path, const bool& dir, const bool& verbose) {
    //outpput vector of Prediction
    std::vector<Prediction> output;
    if(!dir) {
        //load image
        cv::Mat img = load_image(path);
        //perform detection and add to vector
        Prediction pred = detect(img, verbose);
        output.push_back(pred);
    } else {
        //load images
        std::vector<cv::Mat> images = load_images(path);
        //detection and fill vector
        std::vector<Prediction> pred = detect(images, verbose);
        for(int i=0; i<pred.size(); ++i) {
            output.push_back(pred[i]);
        }
    }

    return output;
}

//Prediction class and functions definition

Prediction::Prediction(const cv::Mat& input_image, const std::vector<cv::Rect>& bounding_box, const cv::Mat& output_image) {
    this->input_image = input_image;
    this->bounding_box = bounding_box;
    this->output_image = output_image;
}

bool Prediction::contains_hands() {
    if(bounding_box.size() == 0) {
        return false;
    }
    return true;
}

int Prediction::hands_number() {
    return bounding_box.size();
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
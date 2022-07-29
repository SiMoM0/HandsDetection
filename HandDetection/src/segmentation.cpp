//Implementation of the header "segmentation.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../include/segmentation.hpp"
#include "../include/detection.hpp"
#include "../include/utils.hpp"

Segmenter::Segmenter(Prediction& predicted_image, std::string path_mask) {
    masked_image = cv::Mat::zeros(predicted_image.get_input().size(), CV_8U);
    output_image = predicted_image.get_input().clone();
    bounding_boxes = predicted_image.get_bbox();
    getBoxRegion();
    true_mask = cv::imread(path_mask, CV_8UC1);
}

void Segmenter::segmentRegions() {
    for (int i = 0; i < hand_regions.size(); i++) {
        /*
        cv::Mat hsv;
        cv::Mat tmp(hand_regions[i].rows, hand_regions[i].cols, CV_8UC3);
        cv::cvtColor(hand_regions[i], hsv, cv::COLOR_BGR2HSV);
        cv::Mat hsv_split[3];
        cv::split(hsv, hsv_split);
        otsuSegmentation(hsv_split[2], tmp, 7);
        hand_segmented.push_back(tmp);
        */
        /*
        cv::Mat tmp;
        hsvSegmentation(hand_regions[i],tmp);
        hand_segmented.push_back(tmp);
        */
        /*
        cv::Mat tmp, blur;
        cv::namedWindow("tmp");
        
        multicolorSegmentation(hand_regions[i],tmp);
        blurMask(hand_regions[i], tmp, blur);
        otsuSegmentation(blur,tmp, 5);

        cv::Mat sharp;
        double sigma = 1, threshold = 7, amount = 4;
        cv::Mat low_contr = abs(hand_regions[i] - blur) < threshold;
        sharp = hand_regions[i] * (1 + amount) + blur * (-amount);
        cv::imshow("sharp",sharp);
        otsuSegmentation(sharp,tmp,5);

        int dilation_size = 2;
        cv::Mat dilat, eroded;
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                    cv::Point( dilation_size, dilation_size ) );
        cv::erode(tmp, tmp, element);
        //cv::dilate(eroded, tmp, element);
        

        hand_segmented.push_back(tmp);

        */
    
        cv::Mat tmp;
        multicolorSegmentation(hand_regions[i],tmp);
        //checkImage(hand_regions[i],tmp);
        hand_segmented.push_back(tmp);
        

    /*
       cv::Mat otsu, toret;
       otsuSegmentation(hand_regions[i],otsu, 5);
       kmeansSegmentation(otsu, toret, 2);
       cv::imshow("ret", toret);
       cv::waitKey(0);
       hand_segmented.push_back(toret);
*/
    }
}

void Segmenter::writeSegmented() {
    masked_image = cv::Mat::zeros(output_image.size(), CV_8UC1);
    for (int i = 0; i < hand_segmented.size(); i++) {

        cv::Mat tmp(hand_regions[i]);

        for (int j = 0; j < tmp.rows; j++) {
            for (int z = 0; z < tmp.cols; z++) {
                if (hand_segmented[i].at<uchar>(j,z) != 0) {
                    output_image.at<cv::Vec3b>(bounding_boxes[i].y + j,bounding_boxes[i].x + z) = COL[i];

                    masked_image.at<uchar>(bounding_boxes[i].y + j,bounding_boxes[i].x + z) = 255;
                }
            }
        }
    }
    
}

float Segmenter::pixelAccuracy() {
    int TH = 0, FH = 0;
    
    for (unsigned int i = 0; i < true_mask.rows; i++) {
        for (unsigned int j = 0; j < true_mask.cols; j++) {
            uchar true_val = true_mask.at<uchar>(i,j);
            uchar class_val = masked_image.at<uchar>(i,j);
            if (true_val == 255 && class_val == 255) {
                TH++;
            }
            if (true_val == 0 && class_val == 0) {
                FH++;
            }
        }
    }
    float result = 0;
    result = (static_cast<float>(TH) + static_cast<float>(FH)) / static_cast<float>(true_mask.rows * true_mask.cols);
    return result;
}

void Segmenter::printResults(const int& counter) {
    if (counter == 0) {
        std::cout<<"PERFORMING SEGMENTATION ON IMAGE"<<std::endl;
    }
    std::cout<<"IMAGE "<<counter+1<<std::endl;
    std::cout<<"Pixel accuracy value: "<<pixelAccuracy()<<std::endl;
    std::string title = "Segmented image " + std::to_string(counter+1);
    cv::namedWindow(title);
    cv::imshow(title, output_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    std::cout<<std::endl;
}


void Segmenter::getBoxRegion() {
    for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
        if (bounding_boxes[i].x + bounding_boxes[i].width > output_image.cols) {
            bounding_boxes[i].width = output_image.cols - bounding_boxes[i].x;
        } else if (bounding_boxes[i].y + bounding_boxes[i].height > output_image.rows) {
            bounding_boxes[i].height = output_image.rows - bounding_boxes[i].y;
        }
        hand_regions.push_back(cv::Mat(output_image, bounding_boxes[i]));
    }
}

void hsvSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat image;
    // Maybe remove
    // Used to sharpening the image
    cv::GaussianBlur(input, image, cv::Size(0, 0), 3);
    cv::addWeighted(input, 1.5, image, -0.5, 0, image);

    // Convert to hsv the image
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Get mask in range of hue values
    // Hue in range between 0 and 179
    // Saturation and value between 0 and 255
    cv::Mat mask, tmp;
    cv::inRange(hsv, cv::Scalar(13, 20, 0), cv::Scalar(34.1, 200, 255), tmp);
    mask = (tmp != 0);
    
    // Split the image in 3 channels
    cv::Mat rgb[3];
    cv::split(hsv, rgb);
    for (int i = 0; i < input.channels(); i++) {
        cv::bitwise_and(rgb[i], mask, rgb[i]);
    }
    std::vector<cv::Mat> dest(std::begin(rgb), std::end(rgb));
    cv::merge(dest, output);
    
    
    //cv::cvtColor(output, output, cv::COLOR_HSV2BGR);

}

void regionGrowing(const cv::Mat& input, cv::Mat& mask, const int ksize, uchar similarity) {
    //number of rows and columns of input and output image
    int rows = input.rows;
    int cols = input.cols;
    //predicate Q for controlling growing, 0 if not visited yet, 255 otherwise
    cv::Mat Q = cv::Mat::zeros(rows, cols, CV_8UC1);

    //convert to grayscale, apply blur and threshold (inverse to obtain white for cracks)
    cv::Mat gray, img;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    multicolorSegmentation(input, img);
    //cv::imshow("Threshold", img);

    //loop threshold image to erode pixel groups in a single one (there may be better methods (?))
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            //if the current pixel is black pass this iteration
            if (img.at<uchar>(i, j) == 0)
                continue;
            //flag for controls on neighbors
            bool flag = false;
            //check right, down, left and up pixel, in this order
            if(j < cols-1 && img.at<uchar>(i, j+1) == 255) {
                flag = true;
            } else if(i < rows-1 && img.at<uchar>(i+1, j) == 255) {
                flag = true;
            } else if(j > 0 && img.at<uchar>(i, j-1) == 255) {
                flag = true;
            } else if(i > 0 && img.at<uchar>(i-1, j) == 255) {
                flag = true;
            }

            //change color if flag is true after checking all neighbors
            if(flag)
                img.at<uchar>(i, j) = 0;
        }
    }

    //cv::imshow("Erosion", img);
    //cv::waitKey(0);

    //point to be visited
    std::vector<std::pair<int, int>> points;

    int p = 0;
    //add points of the skeleton image into the vector
    for(int i=0; i<img.rows; ++i) {
        for(int j=0; j<img.cols; ++j) {
            if(img.at<uchar>(i, j) == 255) {
                //add to points vector
                //NOTE: not all the points of the skeleton image may be added, since they could be too much
                points.push_back(std::pair<int, int>(i, j));
                //std::printf("White point at (%d, %d)\n", i, j);
                //update point counter
                p++;
            }
        }
    }
    while(!points.empty()) {
        //pop a single point
        std::pair<int, int> p = points.back();
        points.pop_back();

        //get color value of the point
        uchar color = gray.at<uchar>(p.first, p.second);
        //set the current pixel visited
        Q.at<uchar>(p.first, p.second) = 255;

        //loop for each neighbour
        for(int i=p.first-1; i<=p.first+1; ++i) {
            for(int j=p.second-1; j<=p.second+1; ++j) {
                //check if pixel coordinates exist
                if(i>=0 && i<rows && j>=0 && j<cols) {
                    //get neighbour pixel value
                    uchar neigh = gray.at<uchar>(i, j);
                    //check if it has been visited
                    uchar visited = Q.at<uchar>(i, j);

                    //check if the neighbour pixel is similar
                    if(!visited && std::abs(neigh-color) <= similarity) {
                        points.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
        }
    }
    //copy Q into mask
    mask = Q.clone();
}


void ycbSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat image;
    
    // Convert to YCrCb the image
    cv::Mat ycb; 	
    cv::cvtColor(input, ycb, cv::COLOR_BGR2YCrCb);

    //split brg into 3 channels
    cv::Mat rgb[3];
    cv::split(input, rgb);

    //split ycrcb into 3 channels
    cv::Mat ycrcbChan[3];
    cv::split(ycb, ycrcbChan);

    //set a treshold
    int thresh = 70;
    cv::Scalar threshold(thresh, thresh, thresh);

    cv::Mat mask, minYCB, maxYCB, tmp;
    cv::subtract(ycb, threshold, minYCB);
    cv::add(ycb, threshold, maxYCB);
    

    cv::inRange(ycb, minYCB, maxYCB, tmp);
    mask = (tmp != 0);

    for (int i = 0; i < input.channels(); i++) {
        cv::bitwise_and(rgb[i], mask, rgb[i]);
    }

    std::vector<cv::Mat> dest(std::begin(rgb), std::end(rgb));
    cv::merge(dest, output);

    cv::cvtColor(output, output, cv::COLOR_YCrCb2BGR);
    
    cv::imshow("YCB", output);
}

void bgrSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Vec3b bgrPixel(235, 198, 179);

    cv::Mat3b bgr(bgrPixel);

    int thresh = 40;
    cv::Scalar minBGR = cv::Scalar(bgrPixel.val[0] - thresh, bgrPixel.val[1] - thresh, bgrPixel.val[2] - thresh);
    cv::Scalar maxBGR = cv::Scalar((bgrPixel.val[0] + thresh) % 256, (bgrPixel.val[1] + thresh) % 256, (bgrPixel.val[2] + thresh) % 256);
    
    cv::inRange(bgr, minBGR, maxBGR, output);
    cv::bitwise_and(bgr, bgr, output, output);
    cv::imshow("Result BGR", output);
    cv::waitKey(0);
}


// Doesn't work. Fix it
void Dilation(const cv::Mat& src, const cv::Mat& dst, int dilation_elem, int dilation_size) {
    int dilation_type = 0;
    if ( dilation_elem == 0 ) {
        dilation_type = cv::MORPH_RECT; 
    } else if ( dilation_elem == 1 ){ 
        dilation_type = cv::MORPH_CROSS; 
    } else if( dilation_elem == 2) { 
        dilation_type = cv::MORPH_ELLIPSE; 
    }
    cv::Mat element = cv::getStructuringElement( dilation_type,
                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                    cv::Point( dilation_size, dilation_size ) );
    cv::dilate( src, dst, element );
    cv::imshow( "Dilation Demo", dst );
    cv::waitKey(0);
}





void otsuSegmentation(const cv::Mat& input, cv::Mat& output, const int ksize) {
    cv::Mat gray, temp, mask;
    //convert input image to grayscale
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    //apply blur filter
    //cv::blur(input, temp, cv::Size(ksize, ksize));
    //Otsu optimal threshold to output image
    double value = cv::threshold(gray, mask, 100, 230, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::printf("Otsu threshold: %f\n", value);

    //---segment input image with the original colors---
    //since there are only two segment, store two values
    //find the average values of both areas, comparing with otsu mask
    long sum1[3] = {0, 0, 0}, sum2[3] = {0, 0, 0};
    int count1 = 0, count2 = 0;

    //calculate sum of pixel's value in both areas
    for(int i=0; i<input.rows; ++i) {
        for(int j=0; j<input.cols; ++j) {
            
            //white value in otsu threshold
            if(mask.at<unsigned char>(i, j) > 128) {
                sum1[0] += input.at<cv::Vec3b>(i, j)[0];
                sum1[1] += input.at<cv::Vec3b>(i, j)[1];
                sum1[2] += input.at<cv::Vec3b>(i, j)[2];
                count1++;
            } else {
                sum2[0] += input.at<cv::Vec3b>(i, j)[0];
                sum2[1] += input.at<cv::Vec3b>(i, j)[1];
                sum2[2] += input.at<cv::Vec3b>(i, j)[2];
                count2++;
            }
        }
    }
    
    //calculate average
    uchar avg1[3] = {0, 0, 0};
    uchar avg2[3] = {0, 0, 0};
    for(int i=0; i<3; ++i) {
        avg1[i] = sum1[i] / count1;
        avg2[i] = sum2[i] / count2;
        
    }
    //printf("%d, %d, %d\n", sum2[0], sum2[1], sum2[2]);
    //printf("%d\n", count1);
    //printf("%d\n", count2);
    //printf("%d, %d, %d\n", avg2[0], avg2[1], avg2[2]);
    
    //color the two areas
    for(int i=0; i<output.rows; ++i) {
        for(int j=0; j<output.cols; ++j) {
            //white value in otsu threshold
            if(mask.at<unsigned char>(i, j) > 128) {
                output.at<cv::Vec3b>(i, j) = cv::Vec3b(avg1[0], avg1[1], avg1[2]);
            } else {
                output.at<cv::Vec3b>(i, j) = cv::Vec3b(avg2[0], avg2[1], avg2[2]);
            }
        }
    }

    cv::threshold(output, output, value, 255, 0);
    //cv::adaptiveThreshold(output, output,255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 2);
}

void kmeansSegmentation(const cv::Mat& input, cv::Mat& output, const int k, const bool color) {
    //data array for kmeans function, input image need to be converted to array like
    cv::Mat data = input.reshape(1, input.rows * input.cols);
    //convert to 32 float
    data.convertTo(data, CV_32F);
    
    //structures for kmeans function
    std::vector<int> labels;
    cv::Mat1f centers;
    //apply kmeans
    double compactness = cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1), 10, cv::KMEANS_PP_CENTERS, centers);
    std::printf("Compactness: %f\n", compactness);

    //update data array with clusters colors
    for(int i=0; i<data.rows; ++i) {
        data.at<float>(i, 0) = centers(labels[i], 0);
        data.at<float>(i, 1) = centers(labels[i], 1);
        data.at<float>(i, 2) = centers(labels[i], 2);
    }

    //reshape into output image
    output = data.reshape(3, input.rows);
    output.convertTo(output, CV_8UC3);

    //for task 3, segment only the t-shirts (parameter color set to true)
    if(color) {
        //center number for the t-shirts is the fifth
        int index = 4;
        cv::Mat mask;
        //get the cluster color and apply inRange function
        cv::Scalar color (centers(index, 0), centers(index, 1), centers(index, 2));
        cv::inRange(output, color, color, mask);
        //cv::imshow("Mask", mask);
        //cv::waitKey(0);

        //apply mask using copyTo function and temp Mat
        cv::Mat tmp;
        output.copyTo(tmp, mask);
        //cv::imshow("Final", fin);
        //cv::waitKey(0);

        //clone to output image
        output = tmp.clone();
    }
}

void hslSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat lab;
    cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);

    cv::Vec3b labPixel(lab.at<cv::Vec3b>(0,0));
    
    int thresh = 40;

    cv::Scalar min = cv::Scalar(labPixel.val[0] - thresh, labPixel.val[1] - thresh, labPixel.val[2] - thresh);
    cv::Scalar max = cv::Scalar(labPixel.val[0] + thresh, labPixel.val[1] + thresh, labPixel.val[2] + thresh);

    cv::Mat mask;
    cv::inRange(lab, min, max, mask);
    cv::bitwise_and(lab, lab, output, mask);

    cv::imshow("out", output);
    cv::waitKey(0);
}

void htsSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat edges;
    cannyEdge(input, edges, 21, 5);

    int dilation_size = 2;
    cv::Mat dilat;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS,
                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                    cv::Point( dilation_size, dilation_size ) );
    cv::dilate(edges, dilat, element);
    cv::Mat eroded;
    cv::erode(dilat, eroded, element);

    cv::imshow("dilate", eroded);
}

void cannyEdge(const cv::Mat& input, cv::Mat& output, int sz, int kernel_size) {
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::blur(gray, output, cv::Size(sz, sz));
    cv::Canny(output, output, 0, 180, kernel_size);
}



void edgeTraversalAlgorithm(const cv::Mat& input, cv::Mat& output) {
    
}


void watershedSegmentation(const cv::Mat& input, cv::Mat& output) {
    //implementation of watershed algorithm as described in the documentation (NOT GOOD RESULT)
    cv::Mat bw, dist;

    //convert to grayscale, smooth and use threshold
    cv::cvtColor(input, bw, cv::COLOR_BGR2GRAY);
    cv::blur(bw, bw, cv::Size(5, 5));
    cv::threshold(bw, bw, 60, 255, cv::THRESH_BINARY_INV);
    //cv::imshow("Binary image", bw);
    //cv::waitKey(0);

    //perform the distance transofrm algorithm
    cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
    //normalize distance image
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
    //cv::imshow("Distance", dist);
    //cv::waitKey(0);

    //threshold to obtain peaks, markers for cracks
    cv::threshold(dist, dist, 0.5, 1.0, cv::THRESH_BINARY);
    //dilate the dist image
    //cv::dilate(dist, dist, cv::Mat::ones(3, 3, CV_8U));
    //cv::imshow("Dilate", dist);
    //cv::waitKey(0);

    //from each blob create a seed for watershed algorithm
    cv::Mat dist8u, markers8u;
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
    dist.convertTo(dist8u, CV_8U);
    //find total markers
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dist8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //number of contours
    int ncomp = static_cast<int>(contours.size());
    std::printf("Contours: %d\n", ncomp);

    //draw foreground markers
    for(int i=0; i<ncomp; ++i) {
        cv::drawContours(markers, contours, i, cv::Scalar(i+1), -1);
    }
    //draw background markers
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
    markers.convertTo(markers8u, CV_8U, 10);
    //cv::imshow("Markers", markers8u);
    //cv::waitKey(0);

    //apply the watershed algorithm
    cv::Mat result = input.clone();
    cv::watershed(result, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);

    //generate random colors
    cv::RNG rng (12345);
    std::vector<cv::Vec3b> colors;
    for(int i=0; i<ncomp; ++i) {
        uchar b = static_cast<uchar>(rng.uniform(0, 255));
        uchar g = static_cast<uchar>(rng.uniform(0, 255));
        uchar r = static_cast<uchar>(rng.uniform(0, 255));
        //insert new color
        colors.push_back(cv::Vec3b(b, g, r));
    }

    //create output image
    for(int i=0; i<markers.rows; ++i) {
        for(int j=0; j<markers.cols; ++j) {
            int index = markers.at<int>(i, j);
            std::printf("index: %d\n", index);
            if(index > 0 && index <= ncomp) {
                output.at<cv::Vec3b>(i, j) = colors[index-1];
            }
        }
    }
}

void multicolorSegmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat rgb, hsv, ycbcr;
    
    output = cv::Mat(input.size(), CV_8UC1);
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            cv::Vec3b tmp = input.at<cv::Vec3b>(i,j);
            int RGB_sum = tmp[0] + tmp[1] + tmp[2];
            float R = static_cast<float>(tmp[2]) / RGB_sum;
            float G = static_cast<float>(tmp[1]) / RGB_sum;
            float B = static_cast<float>(tmp[0]) / RGB_sum;

            std::vector<float> RGB = {R,G,B};

            float V = *std::max_element(std::begin(RGB), std::end(RGB));

            float S;
            float normalization_factor = V - *std::min_element(std::begin(RGB), std::end(RGB));
            if (V != 0) {
                S = normalization_factor;
            } else {
                S = 0;
            }

            float H;
            if (V == tmp[2]) {
                H = (60 * (G - B)) / normalization_factor;
            } else if (V == tmp[1]) {
                H = 2 + (60 * (B - R)) / normalization_factor;
            } else if (V == tmp[0]){
                H = 4 + (60 * (R - G)) / normalization_factor;
            }

            if (H < 0) {
                H += 360;
            }

            float delta = 20;
            float Y = tmp[2] * 0.299 + tmp[1] * 0.587 * tmp[0] * 0.114;
            float Cr = (tmp[2] - Y) * 0.713 + delta;
            float Cb = (tmp[0] - Y) * 0.564 + delta;
            
            float R_G;
            if (G == 0) {
                R_G = 1.186;
            } else {
                R_G = R / G;
            }

            if (!(R_G > 1.185) &&
                (H >= 0 && H <= 50) || (H >= 335 && H <= 360) &&
                (S >= 0.2 && S <= 0.6) &&
                (Cb > 77 && Cb < 127) &&
                (Cr > 133 && Cr < 173)) {
                    //output.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                    output.at<uchar>(i,j) = 0;
            } else {
                //output.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
                output.at<uchar>(i,j) = 255;
            }

        }
    }
    //checkImage(input,output);
}

void blurMask(const cv::Mat &input, cv::Mat& mask, cv::Mat& output) {
    cv::blur(input, output, cv::Size(7,7));
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (mask.at<cv::Vec3b>(i,j) != cv::Vec3b(255,255,255)) {
                output.at<cv::Vec3b>(i,j) = input.at<cv::Vec3b>(i,j);
            }
        }
    }
}

//Da finire
void blurMaskborder(const cv::Mat &input, cv::Mat& mask, cv::Mat& output) {
    cv::blur(input, output, cv::Size(7,7));
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (mask.at<cv::Vec3b>(i,j) != cv::Vec3b(255,255,255)) {
                output.at<cv::Vec3b>(i,j) = input.at<cv::Vec3b>(i,j);
            }
        }
    }
}

void checkImage(const cv::Mat& input, cv::Mat& mask) {
    int black_value[] = {0,0,0};
    int black_pixels = 0;
    int white_value[] = {0,0,0};
    int white_pixels = 0;
    for (unsigned int i = 0; i < input.rows; i++) {
        for (unsigned int j = 0; j < input.cols; j++) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(i,j);
            if (mask.at<uchar>(i,j) == 0) {
                black_value[0] += pixel[0];
                black_value[1] += pixel[1];
                black_value[2] += pixel[2];
                black_pixels++;
            } else {
                white_value[0] += pixel[0];
                white_value[1] += pixel[1];
                white_value[2] += pixel[2];
                white_pixels++;
            }
        }
    }
    float tmp;
    for (unsigned int i = 0; i < 3; i++) {
        if (black_pixels != 0) {
            tmp = static_cast<float>(black_value[i] / black_pixels);
            black_value[i] = static_cast<int>(tmp);
        } else {
            black_value[i] = 0;
        }
        if (white_value != 0) {
            tmp = static_cast<float>(white_value[i] / white_pixels);
            white_value[i] = static_cast<int>(tmp);
        } else {
            white_value[i] = 0;
        }
    }
    std::cout<<"Ok"<<std::endl;

    float mse_b = 0, mse_w = 0;
    int skin_color[3] = {218, 231, 250};

    for (unsigned int i = 0; i < 3; i++) {
        mse_b += static_cast<float>(pow(black_value[i] - skin_color[i],2));
        mse_w += static_cast<float>(pow(white_value[i] - skin_color[i],2));
    }
    std::cout<<"Ok"<<std::endl;

    if (mse_b < mse_w) {
        cv::bitwise_not(mask, mask);
    }

}


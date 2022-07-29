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
    get_box_region();
    true_mask = cv::imread(path_mask, CV_8UC1);
}

void Segmenter::segment_regions() {
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
        multicolor_segmentation(hand_regions[i],tmp);
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

void Segmenter::write_segmented() {
    // initialize the mask of the original image
    masked_image = cv::Mat::zeros(output_image.size(), CV_8UC1);
    for (int i = 0; i < hand_segmented.size(); i++) {

        // for each pixel in the selected region
        for (int j = 0; j < hand_regions[i].rows; j++) {
            for (int z = 0; z < hand_regions[i].cols; z++) {
                // if the mask of the region is not at 0
                if (hand_segmented[i].at<uchar>(j,z) != 0) {
                    // set pixel in the output image at specified color
                    // all the pixel selected in the same region shares the same color
                    output_image.at<cv::Vec3b>(bounding_boxes[i].y + j,bounding_boxes[i].x + z) = COL[i];
                    // set the pixel in the masked image at 255, in the end the non selected are all 0
                    masked_image.at<uchar>(bounding_boxes[i].y + j,bounding_boxes[i].x + z) = 255;
                }
            }
        }
    }
    
}

float Segmenter::pixel_accuracy() {
    // TH = True hand correctly segmented counter
    // FH = Non hand correctly segmented counter
    int TH = 0, FH = 0;
    
    // for every pixel in the image update counters
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
    // compute result based on the formula provided as follow
    float result = (static_cast<float>(TH) + static_cast<float>(FH)) / static_cast<float>(true_mask.rows * true_mask.cols);
    return result;
}

void Segmenter::print_results(const int& counter) {
    // if first image selected print the title of the section
    if (counter == 0) {
        std::cout<<"PERFORMING SEGMENTATION ON IMAGE"<<std::endl;
    }

    // print image number
    std::cout<<"IMAGE "<<counter+1<<std::endl;

    // print pixel accuracy
    std::cout<<"Pixel accuracy value: "<<pixel_accuracy()<<std::endl;

    // print output image
    std::string title = "Segmented image " + std::to_string(counter+1);
    cv::namedWindow(title);
    cv::imshow(title, output_image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout<<std::endl;
}


void Segmenter::get_box_region() {
    // retrive from each bounding box the RoI
    for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
        if (bounding_boxes[i].x + bounding_boxes[i].width > output_image.cols) {
            bounding_boxes[i].width = output_image.cols - bounding_boxes[i].x;
        } else if (bounding_boxes[i].y + bounding_boxes[i].height > output_image.rows) {
            bounding_boxes[i].height = output_image.rows - bounding_boxes[i].y;
        }
        hand_regions.push_back(cv::Mat(output_image, bounding_boxes[i]));
    }
}

void hsv_segmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat image;

    // blur image and add weight to it
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

    // apply mask to each channel
    for (int i = 0; i < input.channels(); i++) {
        cv::bitwise_and(rgb[i], mask, rgb[i]);
    }

    // merge the resulted channels
    std::vector<cv::Mat> dest(std::begin(rgb), std::end(rgb));
    cv::merge(dest, output);
}

void region_growing(const cv::Mat& input, cv::Mat& mask, uchar similarity) {
    //number of rows and columns of input and output image
    int rows = input.rows;
    int cols = input.cols;
    //predicate Q for controlling growing, 0 if not visited yet, 255 otherwise
    cv::Mat Q = cv::Mat::zeros(rows, cols, CV_8UC1);

    //convert to grayscale, apply blur and threshold (inverse to obtain white for cracks)
    cv::Mat gray, img;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    multicolor_segmentation(input, img);

    //loop threshold image to erode pixel groups in a single one
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

    //point to be visited
    std::vector<std::pair<int, int>> points;

    int p = 0;
    //add points of the skeleton image into the vector
    for(int i=0; i<img.rows; ++i) {
        for(int j=0; j<img.cols; ++j) {
            if(img.at<uchar>(i, j) == 255) {
                points.push_back(std::pair<int, int>(i, j));
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


void ycbcr_segmentation(const cv::Mat& input, cv::Mat& output) {
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
    // create upper bound and lower bound matrices
    cv::subtract(ycb, threshold, minYCB);
    cv::add(ycb, threshold, maxYCB);
    
    // create the mask in range
    cv::inRange(ycb, minYCB, maxYCB, tmp);
    mask = (tmp != 0);

    // apply the mask to the input image
    for (int i = 0; i < input.channels(); i++) {
        cv::bitwise_and(rgb[i], mask, rgb[i]);
    }

    // merge the channels
    std::vector<cv::Mat> dest(std::begin(rgb), std::end(rgb));
    cv::merge(dest, output);

    // convert the result to BRG 
    cv::cvtColor(output, output, cv::COLOR_YCrCb2BGR);
}

void bgr_segmentation(const cv::Mat& input, cv::Mat& output) {
    // skin color values
    cv::Vec3b bgrPixel(235, 198, 179);
    cv::Mat3b bgr(bgrPixel);

    // apply threshold
    int thresh = 40;
    cv::Scalar minBGR = cv::Scalar(bgrPixel.val[0] - thresh, bgrPixel.val[1] - thresh, bgrPixel.val[2] - thresh);
    cv::Scalar maxBGR = cv::Scalar((bgrPixel.val[0] + thresh) % 256, (bgrPixel.val[1] + thresh) % 256, (bgrPixel.val[2] + thresh) % 256);
    
    // compute the mask
    cv::inRange(bgr, minBGR, maxBGR, output);
    cv::bitwise_and(bgr, bgr, output, output);
}

void otsu_segmentation(const cv::Mat& input, cv::Mat& output, const int ksize) {
    cv::Mat gray, temp, mask;
    //convert input image to grayscale
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    //apply blur filter
    cv::blur(input, temp, cv::Size(ksize, ksize));
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

    // apply a threshold to get the mask
    cv::threshold(output, output, value, 255, 0);
}

void kmeans_segmentation(const cv::Mat& input, cv::Mat& output, const int k, const bool color) {
    //data array for kmeans function, input image need to be converted to array like
    cv::Mat data = input.reshape(1, input.rows * input.cols);
    //convert to 32 float
    data.convertTo(data, CV_32F);
    
    //structures for kmeans function
    std::vector<int> labels;
    cv::Mat1f centers;
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

    if(color) {
        int index = 4;
        cv::Mat mask;
        //get the cluster color and apply inRange function
        cv::Scalar color (centers(index, 0), centers(index, 1), centers(index, 2));
        cv::inRange(output, color, color, mask);

        //apply mask using copyTo function and temp Mat
        cv::Mat tmp;
        output.copyTo(tmp, mask);

        //clone to output image
        output = tmp.clone();
    }
}

void hsl_segmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat lab;
    // convert to LAB 
    // L – Lightness ( Intensity ). 
    // a – color component ranging from Green to Magenta. 
    // b – color component ranging from Blue to Yellow
    cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
    cv::Vec3b labPixel(lab.at<cv::Vec3b>(0,0));
    
    // apply threshold
    int thresh = 40;
    cv::Scalar min = cv::Scalar(labPixel.val[0] - thresh, labPixel.val[1] - thresh, labPixel.val[2] - thresh);
    cv::Scalar max = cv::Scalar(labPixel.val[0] + thresh, labPixel.val[1] + thresh, labPixel.val[2] + thresh);

    // get mask
    cv::Mat mask;
    cv::inRange(lab, min, max, mask);
    cv::bitwise_and(lab, lab, output, mask);
}

void canny_edge(const cv::Mat& input, cv::Mat& output, int sz, int kernel_size) {
    // apply canny to the image
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::blur(gray, output, cv::Size(sz, sz));
    cv::Canny(output, output, 0, 180, kernel_size);
}

void watershed_segmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat bw, dist;

    //convert to grayscale, smooth and use threshold
    cv::cvtColor(input, bw, cv::COLOR_BGR2GRAY);
    cv::blur(bw, bw, cv::Size(5, 5));
    cv::threshold(bw, bw, 60, 255, cv::THRESH_BINARY_INV);

    //perform the distance transofrm algorithm
    cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
    //normalize distance image
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

    //threshold to obtain peaks, markers for cracks
    cv::threshold(dist, dist, 0.5, 1.0, cv::THRESH_BINARY);

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

void multicolor_segmentation(const cv::Mat& input, cv::Mat& output) {
    cv::Mat rgb, hsv, ycbcr;

    // initialize mask output
    output = cv::Mat(input.size(), CV_8UC1);

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            // for each pixel in the input image
            cv::Vec3b tmp = input.at<cv::Vec3b>(i,j);

            // compute the sum of the three channels values
            int RGB_sum = tmp[0] + tmp[1] + tmp[2];

            // compute the normalized value for each channel
            float R = static_cast<float>(tmp[2]) / RGB_sum;
            float G = static_cast<float>(tmp[1]) / RGB_sum;
            float B = static_cast<float>(tmp[0]) / RGB_sum;

            std::vector<float> RGB = {R,G,B};

            // set V of HSV equals to the max of RGB normalized
            float V = *std::max_element(std::begin(RGB), std::end(RGB));

            float S;
            // norm factor equals to V - min of RGB normalized
            float normalization_factor = V - *std::min_element(std::begin(RGB), std::end(RGB));

            // compute V
            if (V != 0) {
                S = normalization_factor;
            } else {
                S = 0;
            }

            // compute H
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

            // compute YCbCr valus
            // delta could be changed
            float delta = 0;
            float Y = tmp[2] * 0.299 + tmp[1] * 0.587 * tmp[0] * 0.114;
            float Cr = (tmp[2] - Y) * 0.713 + delta;
            float Cb = (tmp[0] - Y) * 0.564 + delta;
            
            // compute R/G, set to 1.186 if G is equal 0
            float R_G;
            if (G == 0) {
                R_G = 1.186;
            } else {
                R_G = R / G;
            }

            // set values in the mask respect to the following threshold
            if (!(R_G > 1.185) &&
                (H >= 0 && H <= 50) || (H >= 335 && H <= 360) &&
                (S >= 0.2 && S <= 0.6) &&
                (Cb > 77 && Cb < 127) &&
                (Cr > 133 && Cr < 173)) {
                    output.at<uchar>(i,j) = 0;
            } else {
                output.at<uchar>(i,j) = 255;
            }

        }
    }
}

void blur_mask(const cv::Mat &input, cv::Mat& mask, cv::Mat& output) {
    // blur input image
    cv::blur(input, output, cv::Size(7,7));

    // retain only the element of the blurred image that are inside the masked region
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (mask.at<uchar>(i,j) != 255) {
                output.at<cv::Vec3b>(i,j) = input.at<cv::Vec3b>(i,j);
            }
        }
    }
}

void check_image(const cv::Mat& input, cv::Mat& mask) {
    // mean of black region in the mask
    int black_value[] = {0,0,0};
    // black pixel counter
    int black_pixels = 0;
    // mean of black region in the mask
    int white_value[] = {0,0,0};
    // black pixel counter
    int white_pixels = 0;

    // update mean and counter for each pixel
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

    // normalize each value
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
    // mean square error values
    float mse_b = 0, mse_w = 0;
    // skin color values
    int skin_color[3] = {218, 231, 250};

    // compute the mean square error of the two regions
    for (unsigned int i = 0; i < 3; i++) {
        mse_b += static_cast<float>(pow(black_value[i] - skin_color[i],2));
        mse_w += static_cast<float>(pow(white_value[i] - skin_color[i],2));
    }

    // if the mean square error of the black region is smaller then invert the mask
    if (mse_b < mse_w) {
        cv::bitwise_not(mask, mask);
    }

}


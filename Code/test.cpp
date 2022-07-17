#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;

int main() {
    auto model = dnn::readNet("../Model/best.onnx");

    return 0;
}
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

int main() {
	//images attributes
	const float WIDTH = 640.0;
	const float HEIGHT = 640.0;
	
	//define network
	auto model = dnn::readNet("../Model/best.onnx");
	
	//input image
	string path = "../Dataset/rgb/01.jpg";
	Mat img = imread(path);
	imshow("Input image", img);
	waitKey(0);
	cout << img.type() << "\n";
	//convert to blob
	Mat blob;
	dnn::blobFromImage(img, blob, 1./255., Size(WIDTH, HEIGHT), Scalar(), true, false);
	cout << blob.size << endl;
	//pass to the network
	model.setInput(blob);
	vector<Mat> outputs;
    model.forward(outputs, model.getUnconnectedOutLayersNames());
	cout << outputs[0].size << endl;
	cout << outputs[0].at<float>(0, 100, 0) << endl;
	cout << outputs[0].at<float>(0, 100, 1) << endl;
	cout << outputs[0].at<float>(0, 100, 2) << endl;
	cout << outputs[0].at<float>(0, 100, 3) << endl;
	cout << outputs[0].at<float>(0, 100, 4) << endl;
	cout << outputs[0].at<float>(0, 100, 5) << endl;
	
    return 0;
}

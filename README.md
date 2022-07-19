# Human hands detection and segmentation

Project for the course Computer Vision, UniPD 2022.

### Usage

#### Model

* Download our custom Dataset (link) and upload it saved as *custom_dataset.zip* file on your Google Drive, in the main folder *MyDrive*.

* Upload also the [custom_data.yaml](https://github.com/SiMoM0/HandsDetection/blob/master/Model/data.yaml) file in Google Drive as done before. It will be useful for YOLOv5 configuration.

* Open the Jupyter Notebook [handdetection.ipynb](https://github.com/SiMoM0/HandsDetection/blob/master/Model/handdetection.ipynb) on Google Colaboratory and connect your drive in order to run the code properly.

* At the end of the notebook, run the cell to download the model in *.onnx* format as [best.onnx](https://github.com/SiMoM0/HandsDetection/blob/master/Model/best.onnx) and place it in the directory **Model/**. It will be used for the C++ code.

#### References

[YOLOv5](https://github.com/ultralytics/yolov5): object detection architecture and pretrained models use in this project for fine-tuning

Code:
* Python Hand Detection/Segmentation: https://github.com/BlueDi/Hand-Detection

* Tensorflow: https://medium.com/nerd-for-tech/building-an-object-detector-in-tensorflow-using-bounding-box-regression-2bc13992973f

Paper:
* Hand recognition structure: https://www.researchgate.net/publication/282956557_Real_time_finger_tracking_and_contour_detection_for_gesture_recognition_using_OpenCV

Datasets:
* Hand Dataset: https://www.robots.ox.ac.uk/~vgg/data/hands/

* EgoHands: http://vision.soic.indiana.edu/projects/egohands/

* Kaggle Dataset: https://www.kaggle.com/datasets/armannikkhah/hand-dataset

* SynthHands Dataset: https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/SynthHands.htm
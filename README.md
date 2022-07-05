# Tracklet-based crowd abnormality detection

## Abstract
This algorithm is based on the approach presented by H. Mousavi, M. Nabi, H. Kiani, A. Perina and V. Murino in ["Crowd motion monitoring using tracklet-based commotion measure"](https://ieeexplore.ieee.org/document/7351223). As a result of my implementation, the following functions were added: real-time operation and three ways to determine abnormal behavior (by magnitude, direction, and both at once). The performance of the algorithm has also significantly increased. You can read more about the current implementation and its performance in my dissertation.

To determine anomalies, this algorithm analyzes object tracklets. To extract the tracklets, feature point detectors and optical flow algorithms are used. Before starting the algorithm, you can choose one of the following feature detectors: GFTT, FAST, AGAST, SimpleBlob, SIFT, MSER, KAZE, AKAZE. And one of the optical flow algorithms: Lucas-Kanade or Sparse RLOF.

## Video demos on ped1, ped2 and UMN datasets.
<img src="demo/ped1.gif" width="200"/> <img src="demo/ped2.gif" width="200"/> <img src="demo/umn.gif" width="200"/>

## Usage
### Launch
Before launching, install the OpenCV library. An example of using the algorithm is given in main.cpp. To run the algorithm, you must create a class object and pass it:
- the path to the video file;
- selected feature detectors and optical flow;
- method of capturing data (from the video or from a sequence of images).

### Ground truth setup
There are two ways to pass ground truth to the algorithm:
1. *Via a txt file*. To do this, place a txt file with the same name as the video near the video file. The number of lines in the file is equal to the number of video frames. On each line 0 or 1, depending on the presence of an anomaly. (used for UMN dataset analysis)
```
your_directory
   └------video_name.avi
   └------video_name.txt
```
2. *Via a series of bmp images*. To do this, near the video files folder, you need to place a folder with bmp images, where the position of the anomaly is indicated. Folder name: videoFolderName_gt. The name of the bmp files must match the frame number. (used for ped1 and ped2 datasets analysis).
```
your_directory
   |——————Test001
   |        └------001.tif
   |        └------002.tif
   └——————Test001_gt
            └------001.bmp
            └------002.bmp
```

If you have passed ground truth you will be able to use:
```
void exportProbToFile(string output)
void exportGtToFile(string output)
```
These methods create two files. In the first file - the probability of an anomaly behavior on the current frame/patch. In the second file - ground truth for this frame/patch. In this form, the data can be transferred, for example, to a python program that computes various quality metrics: ROC-AUC, PR-AUC, F1-sscore, etc.

Also, to process ground truth, you must check this parameter:
```
useGroundTruth = true
```
### Parameters
The algorithm settings are at the top in .h file. If you will use one of the ped1, ped 2 or UMN datasets. You can take the perfect parameters already selected for them from the following commits:
```
3f2c09070dfa77d5df714de734ef32c1c6b8a8f9 - ped1
1b61b0750cdf4e88d92c2971963ae093fdb4f952 - ped2
004c22ccc86403a91441df1c14bd336551335b2a - UMN
```
```diff
- Attention!
```
Do not checkout these commits, but only get the parameter values, otherwise you will return to the old version of the algorithm, which may look/work different.

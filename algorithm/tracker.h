#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <cmath>

using namespace cv;
using namespace std;

enum Detectors {
    GFTT_Detector = 0,
    FAST_Detector,
    AGAST_Detector,
    SimpleBlob_Detector,
    SIFT_Detector,
    MSER_Detector,
    KAZE_Detector,
    AKAZE_Detector
};

class Tracker
{
public:
    Tracker(const string& filename, int detector);

public:
    void startTracking();
    void stopTracking();

private:
    void generateColors(int colorsAmount);
    bool getNextFrame();
    void calculateOpticalFlow();
    void filterAndDrawPoint();
    bool showResult();
    void setDetector(int detector_enum);
    void detectNewPoint(Mat &frame, int freq);
    void fillPointMat(int blockSize);

private:
    bool running;

private:
    Mat old_frame;
    Mat new_frame;
    Mat new_color_frame;
    Mat point_mat;
    VideoCapture capture;
    Ptr<cv::Feature2D> detector;
    vector<Scalar> colors;
    vector<Point2f> p0;
    vector<Point2f> p1;
    vector<KeyPoint> new_point;
    vector<uchar> status;
    unsigned int frame_count;
};

#endif // TRACKER_H

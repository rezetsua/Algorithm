#ifndef HUMANTRACKER_H
#define HUMANTRACKER_H

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <cmath>
#include <random>
#include <ctime>

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

class FPoint
{
public:
    FPoint();
    FPoint(Point2f point);

    void operator = (const Point2f& point);
    Scalar generateColor();

public:
    Point2f pt;
    int staticCount;
    Scalar color;
};

class HumanTracker
{
public:
    HumanTracker(const string& filename, int detector);

public:
    void startTracking();
    void stopTracking();

private:
    bool getNextFrame();
    void calculateOpticalFlow();
    void filterAndDrawPoint();
    bool showResult(bool stepByStep);
    void setDetector(int detector_enum);
    void detectNewPoint(Mat &frame, int freq);
    void fillPointMat(int blockSize);
    void deleteStaticPoint(int freq);
    void putInfo(string text, int textY);

private:
    bool running;

private:
    Mat old_frame;
    Mat new_frame;
    Mat new_color_frame;
    Mat point_mat;
    Mat lineMask;
    Mat info;
    VideoCapture capture;
    Ptr<cv::Feature2D> detector;
    vector<FPoint> p0;
    vector<FPoint> p1;
    vector<KeyPoint> new_point;
    vector<uchar> status;
    unsigned int frame_count;
};

#endif // HUMANTRACKER_H

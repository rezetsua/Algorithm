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
    FPoint(Point2f point, int originFrame);

    Scalar generateColor();
    void updatePath();
    void updateVelocity();

public:
    Point2f pt;
    int staticCount;
    int originFrameCount;
    bool goodPath;
    bool dirColor;
    Scalar color;
    double instantVelocity;
    double averageVelocity;
    vector<Point2f> path;
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
    void detectNewPoint(Mat &frame, int queue_index);
    void fillPointMat(int blockSize);
    void deleteStaticPoint(int queue_index);
    void putInfo(string text, int textY);
    void addPointToPath(int queue_index);
    void drawPointPath();
    void approximatePath();
    void drawDirection(vector<Point2f> &apx, int index);
    void fillHSV2BGR();
    Scalar cvtAngleToBGR(int angle);
    void mergePointToObject(int queue_index, int chanels);
    void collectPathInfo(int index);
    void showPathInfo(int queue_index);
    void updateMainStream(int queue_index);

private:
    bool running;
    bool showPoint = false;
    bool showPath = false;
    bool showApproximanedPath = false;
    bool showDirection = false;
    bool showMergePoint = true;

private:
    Mat old_frame;
    Mat new_frame;
    Mat new_color_frame;
    Mat point_mat;
    Mat lineMask;
    Mat directionMask;
    Mat mergeMask;
    Mat info;
    Mat mainStream;
    Mat mainStreamCount;
    VideoCapture capture;
    Ptr<cv::Feature2D> detector;
    vector<FPoint> p0;
    vector<Point2f> p1;
    vector<KeyPoint> new_point;
    vector<uchar> status;
    vector<Scalar> angleToBGR;
    unsigned int frame_count;
    int queue_count;
    int deletedGoodPathAmount;
    int goodPathLifeTimeSum;
};

#endif // HUMANTRACKER_H

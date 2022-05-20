#ifndef HUMANTRACKER_H
#define HUMANTRACKER_H

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>

#include <cmath>
#include <random>
#include <ctime>
#include <stdint.h>
#include <fstream>

using namespace cv;
using namespace std;

const int o = 8; // Orientation dimension
const int m = 4; // Magnitude dimension
const int TL = 5; // Tracklet length

const int magnMax = 8; // Tracklet length

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

enum Flow {
    LUCAS_KANADA = 0,
    RLOF
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
    int newHotCount;
    bool goodPath;
    bool dirColor;
    Scalar color;
    double instantVelocity;
    double averageVelocity;
    int averageVelocityCount;
    vector<Point2f> path;
    vector<uint32_t> hot;
};

class Patch
{
public:
    explicit Patch(Point2f pt);
    void updateComm();
    double getIndexWeight(int j, int jmax);
    std::pair<int, int> getLocalIndex(int i);

    bool isEmpty;
    Point2f center;
    vector<int> lbt; // Local Binary Tracklet (o*m*L)
    vector<int> lbtLifeTime;
    double comm; // Commutation
    vector<double> indexToAngle;
    vector<double> indexToMagnitude;
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
    void calculateOpticalFlow(int flow_enum);
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
    void fillAngleToShift();
    void fillCoordinateToPatchID();
    void fillGridMask();
    void fillPatches();
    void fillGroundTruth(string filename);
    Scalar cvtAngleToBGR(int angle);
    void mergePointToObject(int queue_index, int chanels);
    void collectPathInfo(int index);
    void showPathInfo(int queue_index);
    void updateMainStream(int queue_index);
    void trajectoryAnalysis(int queue_index);
    void updateHOT(int queue_index);
    void calcPatchHOT(int queue_index);
    void calcPatchCommotion(int queue_index);
    void showPatchGist(int queue_index);
    void showPatchComm(int queue_index);
    void patchInit(int index);

public:
    bool running;
    bool showPoint = false;
    bool showPath = true;
    bool showApproximatedPath = false;
    bool showDirection = false;
    bool showMergePoint = false;
    bool trajectoryAnalys = false;

    vector<double> prob;
    vector<int> truth;

private:
    Mat old_frame_color;
    Mat old_frame;
    Mat new_frame;
    Mat new_color_frame;
    Mat point_mat;
    Mat lineMask;
    Mat directionMask;
    Mat mergeMask;
    Mat gridMask;
    Mat info;
    Mat mainStream;
    Mat mainStreamCount;
    Mat coordinateToPatchID;
    Mat patchCommMask;
    vector<int> angleToShift;
    VideoCapture capture;
    vector<uchar> status;
    Ptr<cv::Feature2D> detector;
    vector<FPoint> p0;
    vector<Point2f> p1;
    vector<KeyPoint> new_point;
    vector<Scalar> angleToBGR;
    vector<Patch> patches;
    vector<int> groundTruth;
    unsigned int frame_count;
    int queue_count;
    int deletedGoodPathAmount;
    int goodPathLifeTimeSum;
    int normalPointVelocityAmount;
    int abnormalPointVelocityAmount;
    double averageVelocityRatio;
    int averageVelocityRatioCount;
    int abnormalOutliersFlag;
    int xPatchDim;
    int yPatchDim;
    int dataCollectionCount;
    double globalComm;
};

#endif // HUMANTRACKER_H

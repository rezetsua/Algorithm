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
#include <chrono>
#include <stdint.h>
#include <fstream>

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

enum Flow {
    LUCAS_KANADA = 0,
    RLOF
};

enum CaptureMode {
    VIDEO_CAPTURE = 0,
    IMAGE_CAPTURE
};

enum AnomalyType {
    DETECTION = 0, // Binary answer to the question: Is there an anomaly behavior in the frame?
    LOCALIZATION // Visual answer to the question: Where is the anomaly in the frame?
};

enum AnomalyCalcMode {
    MAGNITUDE = 0,
    DIRECTION,
    BOTH
};

const int o = 8; // Orientation dimension
const int m = 16; // Magnitude dimension
const int TL = 5; // Tracklet length
const int xPatchDim = 6;
const int yPatchDim = 4;

const double magnMax = 4;
const double commTreshToShow = 6.0;
const double patchInitWeight = 2;
const bool bigPatchInit = false;
const int lbtLifeTimeDelta = 20;
const bool lbtResetLifeTime = false;

const int queueIteration = 2;
const int anomalyType = AnomalyType::DETECTION;
const int anomalyCalcMode = AnomalyCalcMode::BOTH;

const int waitkeyPause = 30;
const bool showPoint = false;
const bool showPath = true;
const bool showApproximatedPath = false;
const bool showDirection = false;
const bool showMergePoint = false;
const bool trajectoryAnalys = false;
const bool predictPatchLBT = true;

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
    HumanTracker(const string& filename, int flow = LUCAS_KANADA, int detector = GFTT_Detector, int captureMode = VIDEO_CAPTURE);

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
    void fillPointMat(int blockSize); // Marks the areas of the found points
    void deleteStaticPoint(int queue_index);
    void putInfo(string text, int textY);
    void addPointToPath(int queue_index);
    void drawPointPath();
    void approximatePath();
    void drawDirection(vector<Point2f> &apx, int index);
    void fillHSV2BGR(); // Creates an array to convert the angle of the vector to BGR color according to the HSV model
    void fillAngleToShift();
    void fillCoordinateToPatchID();
    void fillGridMask();
    void fillPatches();
    void fillGroundTruthTXT(string filename);
    void fillGroundTruthIMG(string filename);
    Scalar cvtAngleToBGR(int angle);
    void mergePointToObject(int queue_index, int chanels);
    void collectPathInfo(int index);
    void showPathInfo(int queue_index);
    void trajectoryAnalysis(int queue_index);
    void updateHOT(int queue_index);
    void calcPatchHOT(int queue_index);
    void calcPatchCommotion(int queue_index);
    void showPatchGist(int queue_index);
    void showPatchComm(int queue_index);
    void patchInit(int index);
    void printInfo();
    void addAnomalyTitle(Mat &img);

public:
    bool running;
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
    int captureMode;
    int queue_count;
    int flowType;
    bool anomaly = false;

    long long deletedGoodPathAmount = 0;
    long long goodPathLifeTimeSum = 0;
    double computingTimeCost = 0;
    long long usfullPointAmount = 0;
    long long usfullPointCount = 0;
};

#endif // HUMANTRACKER_H

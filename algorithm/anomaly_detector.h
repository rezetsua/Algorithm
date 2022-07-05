#ifndef ANOMALYDETECTOR_H
#define ANOMALYDETECTOR_H

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

// Algorithm parameters
const int o = 8; // Orientation dimension
const int m = 16; // Magnitude dimension
const int TL = 5; // Tracklet length
const int xPatchDim = 6;
const int yPatchDim = 4;
const double magnMax = 4;

const double commPatchTresh = 6.0; // Value which the patch will turn red
const double commFrameTresh = 8.2; // Value which the behavior is considered abnormal

const bool predictPatchLBT = true; // Use patchInit()
const bool bigPatchInit = false; // The neighbors of the patch will be all the patches around, otherwise only vertical and horizontal
const double patchInitWeight = 2; // The coefficient by which characteristic movement patterns from neighboring patches are multiplied
const int lbtLifeTimeDelta = 20; // Lifetime of a single movement pattern
const bool lbtResetLifeTime = false; // After the lbtLifeTimeDelta, the pattern value is deleted, otherwise it is decremented each time it is used

const int anomalyType = AnomalyType::DETECTION;
const int anomalyCalcMode = AnomalyCalcMode::BOTH;
const int queueIteration = 2; // "Threads" amount
const bool useGroundTruth = false; // Get GT from .txt or .bmp and will generate a 2 txt files with the probability of an anomaly on the current frame/patch and GT

// Visualization
const bool showPoint = false;
const bool showPath = true;
const bool showApproximatedPath = false;
const bool showDirection = false;
const bool showMergePoint = false;
const int waitkeyPause = 10;

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
    bool goodPath; // True - trajectory without occlusions
    bool dirColor; // Direction coded by color
    int staticCount; // The number of frames that the point is stationary
    int originFrameCount; // The number of the frame where the point was first detected
    int averageVelocityCount;
    int newHotCount;
    double instantVelocity;
    double averageVelocity;
    Scalar color;
    vector<Point2f> path;
    vector<uint32_t> hot; // Array of 1's indexes in binary arrays which encode the magnitude and orientation
};

class Patch
{
public:
    explicit Patch(Point2f pt);

    void updateComm();
    double getIndexWeight(int j, int jmax);
    std::pair<int, int> getLocalIndex(int i); // Decoded 1's indexes in binary arrays to magnitude and orientation

    bool isEmpty;
    Point2f center;
    vector<int> lbt; // Local Binary Tracklet (o*m*L)
    vector<int> lbtLifeTime; // How many times has each lbt index been used to calculate the commutation
    double comm; // Commutation (more value -> higher probability of an anomaly in the patch)
    vector<double> indexToAngle;
    vector<double> indexToMagnitude;
};

class AnomalyDetector
{
public:
    AnomalyDetector(const string& filename, int flow = LUCAS_KANADA, int detector = GFTT_Detector, int captureMode = VIDEO_CAPTURE);

public:
    void startAnalysis();
    void stopAnalysis();
    void exportProbToFile(string output);
    void exportGtToFile(string output);

private:
    bool getNextFrame();
    void calculateOpticalFlow(int flow_enum);
    void detectNewPoint(Mat &frame, int queue_index);
    void fillPointMat(int blockSize); // Marks the areas of the found points
    void filterAndDrawPoint();
    void deleteStaticPoint(int queue_index);
    void addPointToPath(int queue_index);
    void approximatePath();
    void collectPathInfo(int index);
    Scalar cvtAngleToBGR(int angle);
    void trajectoryAnalysis(int queue_index);
    void updateHOT(int queue_index);
    void calcPatchHOT(int queue_index);
    void patchInit(int index);
    void calcPatchCommotion(int queue_index);
    void mergePointToObject(int queue_index, int chanels);

    // Visualization
    bool showResult(bool stepByStep);
    void showPatchGist(int queue_index);
    void showPatchComm(int queue_index);
    void drawPointPath();
    void drawDirection(vector<Point2f> &apx, int index);
    void addAnomalyTitle(Mat &img);
    void putInfo(string text, int textY);
    void printInfo();
    void showPathInfo(int queue_index);

    // Initialization
    void setDetector(int detector_enum);
    void fillGridMask(); // create grid mask based on the number of patches
    void fillPatches(); // fill patches array
    // Creates an array to convert:
    void fillHSV2BGR(); // angle of the vector -> BGR color according to the HSV model
    void fillAngleToShift(); // angle of the vector -> 1's indexes in binary arrays
    void fillCoordinateToPatchID(); // point's coordinates -> index of the patch to which it belongs
    void fillGroundTruthTXT(string filename); // frame -> anomaly groundtruth
    void fillGroundTruthIMG(string filename); // patch -> anomaly groundtruth

private:
    Mat oldFrame;
    Mat oldFrameColor;
    Mat newFrame;
    Mat newColorFrame;
    Mat pointMat;
    Mat lineMask;
    Mat directionMask;
    Mat mergeMask;
    Mat gridMask;
    Mat info;
    Mat coordinateToPatchID;
    Mat patchCommMask;

    vector<FPoint> p0;
    vector<Point2f> p1;
    vector<KeyPoint> newPoint;
    vector<int> groundTruth;
    vector<Patch> patches;
    vector<uchar> status;
    vector<int> angleToShift;
    vector<Scalar> angleToBGR;
    vector<double> prob; // Probability of anomaly
    vector<int> truth; // Anomaly groundtruth

    VideoCapture capture;
    Ptr<cv::Feature2D> detector;

    int flowType;
    int captureMode;
    unsigned int frameCount;
    int queueCount; // "Multithreading" counter
    bool anomaly = false; // is current frame has an anomaly
    bool running;

    // Info
    long long deletedGoodPathAmount = 0;
    long long goodPathLifeTimeSum = 0;
    double computingTimeCost = 0;
    long long usfullPointAmount = 0;
    long long usfullPointCount = 0;
};

#endif // ANOMALYDETECTOR_H

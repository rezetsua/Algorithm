#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include "tracker.h"

using namespace cv;
using namespace std;

class Measurement
{
public:
    Measurement();

    void ROC(const Mat &probs, const Mat &truth, vector<Point2f> &roc, int N, const float eps = 0.5);
    float AUC(vector<Point2f> &roc);
    void drawCurve(vector<Point2f> &roc, Mat &img, const Scalar &color);
    void exportToFile(vector<double> &input, string output);
    void exportToFile(vector<int> &input, string output);
    void singleShot(const string& filename, int flow = LUCAS_KANADA, int detector = GFTT_Detector, int captureMode = VIDEO_CAPTURE);

public:
    double EER;
};

#endif // MEASUREMENT_H

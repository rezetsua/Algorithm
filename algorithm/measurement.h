#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

class Measurement
{
public:
    Measurement();

    void ROC(const Mat &probs, const Mat &truth, vector<Point2f> &roc, int N, const float eps=1e-1);
    float AUC(vector<Point2f> &roc);
    void drawCurve(vector<Point2f> &roc, Mat &img, const Scalar &color);

public:
    double EER;
};

#endif // MEASUREMENT_H

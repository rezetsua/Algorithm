#include "measurement.h"

Measurement::Measurement()
{

}

void Measurement::ROC(const Mat &probs, const Mat &truth, vector<Point2f> &roc, int N, const float eps)
{
    double errDiff = 1;
    for (int i = 0; i <= N; i++) {
        float thresh = float(N - i) / N;
        float TP = countNonZero((probs >  thresh) & (truth >  eps));
        float TN = countNonZero((probs <= thresh) & (truth <= eps));
        float FP = countNonZero((probs >  thresh) & (truth <= eps));
        float FN = countNonZero((probs <= thresh) & (truth >  eps));
        float FPR = FP / (FP + TN);
        float TPR = TP / (TP + FN);
        double temp = abs(1 - TPR - FPR);
        if (temp < errDiff) {
            errDiff = temp;
            EER = FPR;
        }
        roc.push_back(Point2f(FPR, TPR));
    }
}

float Measurement::AUC(vector<Point2f> &roc)
{
    float _auc = 0.0f;
    for (int i = 0; i < int(roc.size()) - 1; i++) {
        _auc += (roc[i + 1].y + roc[i].y) * (roc[i + 1].x - roc[i].x);
    }
    return _auc * 0.5f;
}

void Measurement::drawCurve(vector<Point2f> &roc, Mat &img, const Scalar &color)
{
    int   N = roc.size();
    float S = float(img.rows) / N;
    Point2f prev;
    for (size_t i = 0; i < roc.size(); i++) {
        Point2f cur(roc[i].x * N * S, (1.0 - roc[i].y) * N * S); // opencv y axis points down
        if (i > 0)
            line(img, prev, cur, color, 1);
        prev = cur;
    }
}

void Measurement::exportToFile(vector<double> &input, string output)
{
    ofstream fout(output);
    fout << fixed;
    fout.precision(3);
    for (int i = 0; i < input.size(); ++i)
        fout << input[i] << endl;
    fout.close();
}

void Measurement::exportToFile(vector<int> &input, string output)
{
    ofstream fout(output);
    for (int i = 0; i < input.size(); ++i)
        fout << input[i] << endl;
    fout.close();
}

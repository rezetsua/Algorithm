#include "tracker.h"
#include "measurement.h"

int main()
{
    // Tracking
    string filename = "/home/urii/Документы/DataSet/scene1.avi";
    HumanTracker tracker(filename, Detectors::GFTT_Detector);
    tracker.startTracking();

    normalize(tracker.prob, tracker.prob, 1, 0, NORM_MINMAX);

    Measurement m;
    vector<Point2f> roc;
    Mat probs(tracker.prob);
    Mat truth(tracker.truth);

    m.ROC(probs, truth, roc, 10000);
    Mat roc_draw(480, 480, CV_8UC3, Scalar::all(255));
    m.drawCurve(roc, roc_draw, Scalar(255, 0, 0));
    line(roc_draw, Point2f(0, 0), Point2f(roc_draw.cols, roc_draw.rows), Scalar(0, 255, 0), 1);
    float auc = m.AUC(roc);
    cv::putText(roc_draw, "AUC " + std::to_string(auc), cv::Point(350, 400), cv::FONT_HERSHEY_DUPLEX, 0.7, Scalar(0,0,255), 2);
    cv::putText(roc_draw, "EER " + std::to_string(m.EER), cv::Point(350, 430), cv::FONT_HERSHEY_DUPLEX, 0.7, Scalar(0,0,255), 2);

    imshow("ROC", roc_draw);
    waitKey(0);

    return 0;
}

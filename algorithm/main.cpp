#include "anomaly_detector.h"

int main()
{
    AnomalyDetector detector("/home/urii/Документы/DataSet/UMN/123scene.avi", Flow::LUCAS_KANADA,
                         Detectors::GFTT_Detector, CaptureMode::VIDEO_CAPTURE);

    detector.startAnalysis();

    // If the algorithm was given a GT
    detector.exportProbToFile("/home/urii/Документы/DataSet/UMN/123scene.txt");
    detector.exportGtToFile("/home/urii/Документы/DataSet/UMN/123scene.txt");

    return 0;
}

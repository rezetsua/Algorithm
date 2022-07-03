#include "tracker.h"

int main()
{
    HumanTracker tracker("/home/urii/Документы/DataSet/UMN/123scene.avi", Flow::LUCAS_KANADA,
                         Detectors::GFTT_Detector, CaptureMode::VIDEO_CAPTURE);

    tracker.startTracking();
    tracker.exportProbToFile("/home/urii/Документы/DataSet/UMN/123sceneProb.txt");
    tracker.exportGtToFile("/home/urii/Документы/DataSet/UMN/123sceneGT.txt");

    return 0;
}

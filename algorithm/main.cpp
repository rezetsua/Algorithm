#include "tracker.h"

int main()
{
    HumanTracker tracker("/home/urii/Документы/DataSet/23scene.mp4", Flow::LUCAS_KANADA,
                         Detectors::GFTT_Detector, CaptureMode::VIDEO_CAPTURE);

    tracker.startTracking();

    // If the algorithm was given a GT
    tracker.exportProbToFile("/home/urii/Документы/DataSet/123sceneProb.txt");
    tracker.exportGtToFile("/home/urii/Документы/DataSet/123sceneGT.txt");

    return 0;
}

#include "tracker.h"

int main()
{
    //string filename = "/home/urii/Документы/DataSet/explosion/4.mp4";
    string filename = "/home/urii/Документы/DataSet/anomaly.avi";
    //string filename = "/home/urii/Документы/DataSet/grandcentral.avi";
    HumanTracker tracker(filename, Detectors::GFTT_Detector);
    tracker.startTracking();

    return 0;
}

#include "tracker.h"

int main()
{
    string filename = "/home/urii/Документы/DataSet/grandcentral.avi";
    Tracker tracker(filename, Detectors::GFTT_Detector);
    tracker.startTracking();

    return 0;
}

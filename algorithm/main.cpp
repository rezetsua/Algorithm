#include "tracker.h"
#include "measurement.h"

int main()
{
    Measurement m;
    // Tracking
//    m.singleShot("/home/urii/Документы/DataSet/UMN/scene3.avi",
//                 Detectors::GFTT_Detector, CaptureMode::VIDEO_CAPTURE);

    fstream clear_file1("/home/urii/Документы/DataSet/txt/proba.txt", ios::out);
    clear_file1.close();
    fstream clear_file2("/home/urii/Документы/DataSet/txt/truth.txt", ios::out);
    clear_file2.close();

    m.singleShot("/home/urii/Документы/DataSet/ped2/Test005/001.tif",
                 Detectors::AKAZE_Detector, CaptureMode::IMAGE_CAPTURE);

    return 0;
}

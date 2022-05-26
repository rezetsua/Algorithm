#include "tracker.h"
#include "measurement.h"

int main()
{
    Measurement m;
    // Tracking

    fstream clear_file1("/home/urii/Документы/DataSet/txt/proba.txt", ios::out);
    clear_file1.close();
    fstream clear_file2("/home/urii/Документы/DataSet/txt/truth.txt", ios::out);
    clear_file2.close();

    m.singleShot("/home/urii/Документы/DataSet/UMN/123scene.avi", Flow::RLOF,
                 Detectors::AKAZE_Detector, CaptureMode::VIDEO_CAPTURE);

//    m.singleShot("/home/urii/Документы/DataSet/ped1/Test019/001.tif",
//                 Detectors::GFTT_Detector, CaptureMode::IMAGE_CAPTURE);

    return 0;
}

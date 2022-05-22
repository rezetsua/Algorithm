#include "tracker.h"
#include "measurement.h"

int main()
{
    Measurement m;
    // Tracking
//    m.singleShot("/home/urii/Документы/DataSet/scene3.avi",
//                 Detectors::GFTT_Detector, CaptureMode::VIDEO_CAPTURE);

    fstream clear_file1("/home/urii/Документы/DataSet/proba.txt", ios::out);
    clear_file1.close();
    fstream clear_file2("/home/urii/Документы/DataSet/truth.txt", ios::out);
    clear_file2.close();

    m.singleShot("/home/urii/Загрузки/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test002/001.tif",
                 Detectors::GFTT_Detector, CaptureMode::IMAGE_CAPTURE);
    m.singleShot("/home/urii/Загрузки/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test005/001.tif",
                 Detectors::GFTT_Detector, CaptureMode::IMAGE_CAPTURE);
    m.singleShot("/home/urii/Загрузки/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test008/001.tif",
                 Detectors::GFTT_Detector, CaptureMode::IMAGE_CAPTURE);

    return 0;
}

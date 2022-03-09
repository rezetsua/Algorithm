#include "tracker.h"

//bool isOldPoint(Point2f &newP, Point2f oldP, int windowSize) {
//    int delta = windowSize / 2;
//    if (newP.x <= oldP.x + delta && newP.x >= oldP.x - delta)
//        if (newP.y <= oldP.y + delta && newP.y >= oldP.y - delta)
//            return true;
//    return false;
//}

//void addNewPoint(vector<Point2f> &newPoint, vector<Point2f> &oldPoint, int windowSize) {
//    for (int i = newPoint.size() - 1; i >= 0; i--) {
//        for (int j = 0; j < oldPoint.size(); j++) {
//            if (isOldPoint(newPoint[i], oldPoint[j], windowSize)) {
//                newPoint.pop_back();
//                break;
//            }
//        }
//    }
//    cout << "size " << newPoint.size() << endl;
//    for (int j = 0; j < newPoint.size(); j++)
//        oldPoint.push_back(newPoint[j]);
//}

////void pointToMat(vector<Point2f> &point, Mat &pointMat) {
////    for (int i = 0; i < point.size(); i++)
////        if (pointMat.at<uchar>(point[i].y, point[i].x))

////}

//int lucas_kanade(const string& filename, bool save)
//{
//    int keyPointAmount = 1000;
//    int windowSize = 10;
//    double keyQuality = 0.01;

//    unsigned int t = 0;  // time in frames


//    VideoCapture capture(filename);
//    if (!capture.isOpened()){
//        //error in opening the video input
//        cerr << "Unable to open file!" << endl;
//        return 0;
//    }
//    // Create some random colors
//    vector<Scalar> colors;
//    RNG rng;
//    for(int i = 0; i < keyPointAmount; i++)
//    {
//        int r = rng.uniform(0, 256);
//        int g = rng.uniform(0, 256);
//        int b = rng.uniform(0, 256);
//        colors.push_back(Scalar(r,g,b));
//    }
//    Mat old_frame, old_gray;
//    vector<Point2f> p0, p1;  // Предыдущие и новые координаты пикселей
//    // Take first frame and find corners in it
//    capture >> old_frame;
//    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);

//    Mat pointMat = Mat::zeros(old_gray.size(), old_gray.type());

//    // Особые точки находятся только 1 раз
////    detector = GFTTDetector::create(3000, 0.01, 1, 3, false, 0.04);
////    create( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
////            int blockSize=3, bool useHarrisDetector=false, double k=0.04 );

////    goodFeaturesToTrack( InputArray image, OutputArray corners,
////                         int maxCorners, double qualityLevel, double minDistance,
////                         InputArray mask = noArray(), int blockSize = 3,
////                         bool useHarrisDetector = false, double k = 0.04 );

//    //cout << p0.size() << endl;
//    // Create a mask image for drawing purposes
//    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

//    goodFeaturesToTrack(old_gray, p0, keyPointAmount, keyQuality, windowSize, Mat(), 7, false, 0.04);

//    while(true){
//        t++;

//        Mat frame, frame_gray;
//        capture >> frame;
//        if (frame.empty())
//            break;
//        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

//        if (t % 30 == 0) {
//            vector<Point2f> newPoint;
//            goodFeaturesToTrack(frame_gray, newPoint, keyPointAmount, keyQuality, windowSize, Mat(), 7, false, 0.04);
//            cout << "p0 size " << p0.size() << " newPoint size " << newPoint.size() << endl;
//            double t = (double)getTickCount();
//            addNewPoint(newPoint, newPoint, 40);
//            t = ((double)getTickCount() - t)/getTickFrequency();
//            cout << "t = " << t << endl;
//        }

//        // calculate optical flow
//        vector<uchar> status;
//        vector<float> err;
//        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
//        /* status -  1 если для точки построен поток, 0 если нет
//         * предпоследнее - OPTFLOW_LK_GET_MIN_EIGENVALS
//         * последнее - пороговое значение для фильтрации точек
//         */
//        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria, 0, 1e-4 );
//        vector<Point2f> good_new;
//        int count = 0;
//        //cout << "size " << p0.size() << endl;
//        for(uint i = 0; i < p0.size(); i++)
//        {
//            // Select good points
//            if (status[i] == 0)
//                continue;
//            good_new.push_back(p1[i]);
//            float deltaX = abs(p1[i].x - p0[i].x);
//            float deltaY = abs(p1[i].y - p0[i].y);
//            //cout << "deltaX " << deltaX << "deltaY " << deltaY << endl;
//            // draw the tracks
//            float treshold = 0.1;
//            if (deltaX < treshold && deltaY < treshold) {
//                count++;
//                continue;
//            }
//            //line(mask,p1[i], p0[i], colors[i], 2);
//            circle(frame, p1[i], 5, Scalar(0,200,0), -1);
//        }
//        //cout << "count" << count << endl;
//        Mat img;
//        add(frame, mask, img);
//        imshow("flow", img);
//        int keyboard = waitKey(30);
//        if (keyboard == 'q' || keyboard == 27)
//            break;
//        // Now update the previous frame and previous points
//        old_gray = frame_gray.clone();
//        p0 = good_new;
//    }
//    waitKey(0);
//    return 0;
//}

int main()
{
    string filename = "/home/urii/Документы/DataSet/grandcentral.avi";
    Tracker tracker(filename, Detectors::MSER_Detector);
    tracker.startTracking();

    return 0;
}

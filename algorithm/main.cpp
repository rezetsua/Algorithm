#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/optflow.hpp>

#include <cmath>

using namespace cv;
using namespace std;

int lucas_kanade(const string& filename, bool save)
{
    VideoCapture capture(filename);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;  // Предыдущие и новые координаты пикселей
    // Take first frame and find corners in it
    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    // Особые точки находятся только 1 раз
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    while(true){
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        // calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        /* status -  1 если для точки построен поток, 0 если нет
         * предпоследнее - OPTFLOW_LK_GET_MIN_EIGENVALS
         * последнее - пороговое значение для фильтрации точек
         */
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria, 0, 1e-4 );
        vector<Point2f> good_new;
        int count = 0;
        //cout << "size " << p0.size() << endl;
        for(uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 0)
                continue;
            good_new.push_back(p1[i]);
            float deltaX = abs(p1[i].x - p0[i].x);
            float deltaY = abs(p1[i].y - p0[i].y);
            //cout << "deltaX " << deltaX << "deltaY " << deltaY << endl;
            // draw the tracks
            float treahold = 0.2;
            if (deltaX < treahold && deltaY < treahold) {
                count++;
                continue;
            }
            line(mask,p1[i], p0[i], colors[i], 2);
            circle(frame, p1[i], 5, colors[i], -1);
        }
        //cout << "count" << count << endl;
        Mat img;
        add(frame, mask, img);
        imshow("flow", img);
        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;
        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
    return 0;
}

int main()
{
    lucas_kanade("/home/urii/Документы/DataSet/grandcentral.avi", false);

    return 0;
}

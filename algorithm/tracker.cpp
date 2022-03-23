#include "tracker.h"

Tracker::Tracker(const string& filename, int detector_enum)
{
    capture.open(filename);
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;

    generateColors(2000);

    capture >> old_frame;
    frame_count = 1;
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);

    point_mat = Mat::zeros(old_frame.size(), old_frame.type());
    imshow("black", point_mat);

    setDetector(detector_enum);
    detectNewPoint(old_frame, 1);
}

void Tracker::stopTracking()
{
    running = false;
}

void Tracker::startTracking()
{
    running = true;

    while(running){

        if (!getNextFrame()) break;

        detectNewPoint(new_frame, 10);

        calculateOpticalFlow();

        filterAndDrawPoint();

        if (!showResult()) break;
    }

    waitKey(0);
}

void Tracker::generateColors(int colorsAmount)
{
    RNG rng;
    for(int i = 0; i < colorsAmount; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
}

bool Tracker::getNextFrame()
{
    capture >> new_color_frame;
    if (new_color_frame.empty())
        return false;
    cvtColor(new_color_frame, new_frame, COLOR_BGR2GRAY);
    frame_count++;
    return true;
}

void Tracker::calculateOpticalFlow()
{
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(old_frame, new_frame, p0, p1, status, err, Size(15,15), 2, criteria, 0, 1e-4 );
}

void Tracker::filterAndDrawPoint()
{
    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Filter
        if (status[i] == 0)
            continue;
        good_new.push_back(p1[i]);
        float deltaX = abs(p1[i].x - p0[i].x);
        float deltaY = abs(p1[i].y - p0[i].y);
        float treshold = 0.1;
        if (deltaX < treshold && deltaY < treshold) {
            continue;
        }
        // Draw
        circle(new_color_frame, p1[i], 5, Scalar(0,200,0), -1);
    }
    p0 = good_new;
}

bool Tracker::showResult()
{
    imshow("flow", new_color_frame);
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
        return false;
    old_frame = new_frame.clone();
    return true;
}

void Tracker::setDetector(int detector_enum)
{
    switch (detector_enum) {
    case GFTT_Detector: {
        // Увеличить количество точек: maxCorners+
        detector = GFTTDetector::create(1000, 0.1, 7, 3, false, 0.04);
        break;
    }
    case FAST_Detector: {
        // Увеличить количество точек: threshold-, type = TYPE_9_16
        detector = FastFeatureDetector::create(20, true, FastFeatureDetector::TYPE_9_16);
        break;
    }
    case AGAST_Detector: {
        // Увеличить количество точек: threshold-, type = OAST_9_16
        detector = AgastFeatureDetector::create(14, true, AgastFeatureDetector::AGAST_5_8);
        break;
    }
    case SimpleBlob_Detector: {
        // Увеличить количество точек: minArea-
        // Setup SimpleBlobDetector parameters.
        SimpleBlobDetector::Params params;
        // Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 200;
        // Filter by Area.
        params.filterByArea = true;
        params.minArea = 5;
        // Filter by Circularity
        params.filterByCircularity = false;
        params.minCircularity = 0.1;
        // Filter by Convexity
        params.filterByConvexity = false;
        params.minConvexity = 0.87;
        // Filter by Inertia
        params.filterByInertia = false;
        params.minInertiaRatio = 0.01;
        detector = SimpleBlobDetector::create(params);
        break;
    }
//    case SIFT_Detector: {
//        detector = SIFT::create(0, 3, 0.04, 10, 1.6);
//        break;
//    }
    case MSER_Detector: {
        /*@param _delta it compares \f$(size_{i}-size_{i-delta})/size_{i-delta}\f$
          @param _min_area prune the area which smaller than minArea
          @param _max_area prune the area which bigger than maxArea
          @param _max_variation prune the area have similar size to its children
          Увеличить количество точек: delta-, max_variation+ */
        detector = MSER::create(20, 80, 200, 0.5);;
        break;
    }
    case KAZE_Detector: {
        // Увеличить количество точек: threshold-, diffusivity = DIFF_CHARBONNIER
        detector = KAZE::create(false, false, 0.0005f, 4, 4, KAZE::DIFF_PM_G2);
        break;
    }
    case AKAZE_Detector: {
        // Увеличить количество точек: threshold-, diffusivity = DIFF_CHARBONNIER
        detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 4, 4, KAZE::DIFF_PM_G2);
        break;
    }
    }
}

void Tracker::detectNewPoint(Mat &frame, int freq)
{
    double t = (double)getTickCount();
    if (frame_count % freq != 0)
        return;

    detector->detect(frame, new_point);

    fillPointMat(7);
    for (int i = 0; i < new_point.size(); i++) {
        if (point_mat.at<uchar>(new_point[i].pt.y, new_point[i].pt.x) == 255)
            continue;
        p0.push_back(new_point[i].pt);
        circle(point_mat, new_point[i].pt, 7, Scalar(255), -1);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << t << endl;
    cout << "p0 size" << p0.size() << endl;
    imshow("black", point_mat);
}

void Tracker::fillPointMat(int blockSize)
{
    point_mat = Mat::zeros(old_frame.size(), old_frame.type());
    for (int i = 0; i < p0.size(); i++) {
        circle(point_mat, p0[i], blockSize, Scalar(255), -1);
    }
}

#include "tracker.h"

Tracker::Tracker(const string& filename, int detector_enum)
{
    capture.open(filename);
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;

    generateColors(2000);

    capture >> old_frame;
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);

    point_mat = Mat::zeros(old_frame.size(), old_frame.type());

    vector<KeyPoint> keypoint;
    setDetector(detector_enum);
    detector->detect(old_frame, keypoint);

    vector<Point2f> temp;
    for (int i = 0; i < keypoint.size(); i++)
        temp.push_back(keypoint[i].pt);
    p0 = temp;
}

void Tracker::stopTracking()
{
    running = false;
}

void Tracker::startTracking()
{
    running = true;

    while(running){
        if (!getNextFrame())
            break;
        calculateOpticalFlow();
        filterAndDrawPoint();
        if (!showResult())
            break;
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
        detector = GFTTDetector::create(3000, 0.01, 1, 3, false, 0.04);
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

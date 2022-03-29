#include "tracker.h"

HumanTracker::HumanTracker(const string& filename, int detector_enum)
{
    capture.open(filename);
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;

    capture >> old_frame;
    frame_count = 1;
    queue_count = 1;

    lineMask = Mat::zeros(old_frame.size(), old_frame.type());
    directionMask = Mat::zeros(old_frame.size(), old_frame.type());
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);

    info = Mat::zeros(old_frame.size(), old_frame.type());
    point_mat = Mat::zeros(old_frame.size(), old_frame.type());

    setDetector(detector_enum);
    detectNewPoint(old_frame, 1);
}

void HumanTracker::stopTracking()
{
    running = false;
}

void HumanTracker::startTracking()
{
    running = true;

    while(running){
        double t = (double)getTickCount();

        if (!getNextFrame()) break;

        detectNewPoint(new_frame, 1);

        calculateOpticalFlow();

        filterAndDrawPoint();

        deleteStaticPoint(2);

        addPointToPath(3);

        t = ((double)getTickCount() - t)/getTickFrequency();
        if (frame_count % 10 == 0)
            putInfo("FPS " + std::to_string((int)(1/t)), 400);

        if (!showResult(false)) break;
    }

    waitKey(0);
}


bool HumanTracker::getNextFrame()
{
    capture >> new_color_frame;
    if (new_color_frame.empty())
        return false;
    cvtColor(new_color_frame, new_frame, COLOR_BGR2GRAY);
    frame_count++;
    queue_count++;
    if (queue_count > 10)
        queue_count = 1;
    return true;
}

void HumanTracker::calculateOpticalFlow()
{
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

    vector<Point2f> point0;
    for (int i = 0; i < p0.size(); i++)
        point0.push_back(p0[i].pt);

    calcOpticalFlowPyrLK(old_frame, new_frame, point0, p1, status, err, Size(15,15), 2, criteria, 0, 1e-4 );
}

void HumanTracker::filterAndDrawPoint()
{
    int count = 0;
    for(int i = 0; i < p1.size(); i++)
    {
        double delta = cv::norm(p1[i] - p0[i].pt);
        float treshold = 0.2;
        if (delta < treshold ) {
            p0[i].staticCount++;
            continue;
        }
        p0[i].staticCount = 0;
        // Draw
        //circle(new_color_frame, p1[i], 2, p0[i].color, -1);
        count++;
    }
    for (int i = 0; i < p1.size(); i++)
        p0[i].pt = p1[i];
}

bool HumanTracker::showResult(bool stepByStep)
{
    int pauseTime = stepByStep ? 0 : 30;

    Mat directionMaskBGR = lineMask.clone();
    cvtColor(directionMask, directionMaskBGR, COLOR_HSV2BGR);
    imshow("directionMaskBGR", directionMaskBGR);
    add(lineMask, directionMaskBGR, lineMask);
    add(new_color_frame, lineMask, new_color_frame);
    imshow("info", info);
    imshow("flow", new_color_frame);
    if (waitKey(pauseTime) == 27)
        return stepByStep;
    old_frame = new_frame.clone();
    return true;
}

void HumanTracker::setDetector(int detector_enum)
{
    switch (detector_enum) {
    case GFTT_Detector: {
        // Увеличить количество точек: maxCorners+
        detector = GFTTDetector::create(1000, 0.07, 7, 3, false, 0.04);
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

void HumanTracker::detectNewPoint(Mat &frame, int queue_index)
{
    if (queue_count != queue_index)
        return;

    detector->detect(frame, new_point);

    fillPointMat(7);
    for (int i = 0; i < new_point.size(); i++) {
        if (point_mat.at<uchar>(new_point[i].pt.y, new_point[i].pt.x) == 255)
            continue;
        p0.push_back(FPoint(new_point[i].pt, frame_count));
        circle(point_mat, new_point[i].pt, 7, Scalar(255), -1);
    }
    putInfo("Total point amount " + std::to_string(p0.size()), 100);
    //imshow("occupied area", point_mat);
}

void HumanTracker::fillPointMat(int blockSize)
{
    point_mat = Mat::zeros(old_frame.size(), old_frame.type());
    for (int i = 0; i < p0.size(); i++) {
        circle(point_mat, p0[i].pt, blockSize, Scalar(255), -1);
    }
}

void HumanTracker::deleteStaticPoint(int queue_index)
{
    if (queue_count != queue_index)
        return;
    int count = 0;
    for (int i = p0.size() - 1; i >= 0; i--) {
        if (p0[i].staticCount >= 5) {
            p0.erase(p0.begin() + i);
            count++;
        }
    }
    putInfo("Deleted static point " + std::to_string(count), 200);
}

void HumanTracker::putInfo(string text, int textY)
{
    Rect rect(10, textY - 80, info.cols, 100);
    rectangle(info, rect, cv::Scalar(0), -1);
    cv::putText(info, text, cv::Point(10, textY), cv::FONT_HERSHEY_DUPLEX, 1.0, Scalar(255), 2);
}

void HumanTracker::addPointToPath(int queue_index)
{
    if (queue_count != queue_index)
        return;
    for (int i = 0; i < p0.size(); i++) {
        p0[i].updatePath();
    }
    approximatePath();
    //drawPointPath();
}

void HumanTracker::drawPointPath()
{
    lineMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    for (int i = 0; i < p0.size(); i++) {
        // Filter
        if (p0[i].path.size() > 2) {
            // Draw
            for (int j = 1; j < p0[i].path.size(); j++)
                line(lineMask, p0[i].path[j], p0[i].path[j - 1],
                     Scalar(0, p0[i].averageVelocity * 20, 255 - p0[i].averageVelocity * 20), 2);
        }
    }
}

void HumanTracker::approximatePath()
{
    lineMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    directionMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    vector<Point2f> apx;
    for (int i = 0; i < p0.size(); i++) {
        if (p0[i].path.size() > 2) {
            // Approximate
            double epsilon = 0.2 * arcLength(p0[i].path, false);
            approxPolyDP(p0[i].path, apx, epsilon, false);
            // Filter
            p0[i].goodPath = apx.size() > 2 ? false : true;
            Scalar color = p0[i].goodPath
                         ? Scalar(0, p0[i].averageVelocity * 20, 255 - p0[i].averageVelocity * 20)
                         : Scalar(255, 0, 0);
            // Draw
//            for (int j = 1; j < apx.size(); j++)
//                line(lineMask, apx[j], apx[j - 1], color, 2);
            if (p0[i].goodPath)
                drawDirection(apx, i);
        }
    }
    // Delete bad point
    int count = 0;
    for (int i = p0.size() - 1; i >= 0; i--) {
        if (!p0[i].goodPath) {
            p0.erase(p0.begin() + i);
            count++;
        }
    }
    putInfo("Deleted bad point " + std::to_string(count), 300);
}

void HumanTracker::drawDirection(vector<Point2f> &apx, int index)
{
    if (apx.size() < 2)
        return;

    Point2f pointDirection;
    //int predictDiv = 2;
    int predictDiv = 4;
    float p0x = apx[apx.size() - 2].x;
    float p1x = apx[apx.size() - 1].x;
    float deltax = p1x - p0x;
    pointDirection.x = p1x + deltax / predictDiv;

    float p0y = apx[apx.size() - 2].y;
    float p1y = apx[apx.size() - 1].y;
    float deltay = p1y - p0y;
    pointDirection.y = p1y + deltay / predictDiv;

    int angle = round(atan2(deltay, deltax) * 180 / 3.14);
    if (angle < 0)
        angle = 360 + angle;

    //Scalar dirColor(angle / 2, p0[index].averageVelocity * 25, 255);
    Scalar dirColor(angle / 2, 255, 255);

    arrowedLine(directionMask, apx[apx.size() - 1], pointDirection, dirColor, 1);
    //circle(directionMask, pointDirection, 3, dirColor, -1);
}

FPoint::FPoint()
{

}

FPoint::FPoint(Point2f point, int originFrame)
{
    pt = point;
    staticCount = 0;
    instantVelocity = 0;
    averageVelocity = 0;
    originFrameCount = originFrame;
    goodPath = true;
    color = generateColor();
}

Scalar FPoint::generateColor()
{
    auto now = std::chrono::high_resolution_clock::now();
    std::mt19937 gen(now.time_since_epoch().count());
    std::uniform_int_distribution<> uid(1, 255);
    int r = uid(gen);
    int g = uid(gen);
    int b = uid(gen);
    return Scalar(r,g,b);
}

void FPoint::updatePath()
{
    if (path.size() < 5)
        path.push_back(pt);
    else {
        path.erase(path.begin());
        path.push_back(pt);
    }
    updateVelocity();
}

void FPoint::updateVelocity()
{
    if (path.size() < 2)
        return;
    instantVelocity = cv::norm(path[path.size() - 1] - path[path.size() - 2]);
    averageVelocity = cv::norm(path[path.size() - 1] - path[0]) / path.size();
}


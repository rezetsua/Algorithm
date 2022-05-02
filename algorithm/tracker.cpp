#include "tracker.h"

HumanTracker::HumanTracker(const string& filename, int detector_enum)
{
    capture.open(filename);
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;

    capture >> old_frame;
    frame_count = 1;
    queue_count = 1;
    deletedGoodPathAmount = 0;
    goodPathLifeTimeSum = 0;
    averageVelocityRatio = 0;
    abnormalOutliersFlag = 0;

    lineMask = Mat::zeros(old_frame.size(), old_frame.type());
    directionMask = Mat::zeros(old_frame.size(), old_frame.type());
    mergeMask = Mat::zeros(old_frame.size(), old_frame.type());
    mainStream = Mat::zeros(old_frame.size(), old_frame.type());
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);

    info = Mat::zeros(480, 480, old_frame.type());
    point_mat = Mat::zeros(old_frame.size(), old_frame.type());
    mainStreamCount = Mat::zeros(old_frame.size(), CV_32S);

    fillHSV2BGR();
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

        trajectoryAnalysis(3);

        mergePointToObject(4, 12);

        showPathInfo(2);

        t = ((double)getTickCount() - t)/getTickFrequency();
        if (frame_count % 10 == 0)
            putInfo("FPS " + std::to_string((int)(1/t)), 5);

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
    if (queue_count > 5)
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
        float treshold = 0.3;
        if (delta < treshold ) {
            p0[i].staticCount++;
            continue;
        }
        p0[i].staticCount = 0;
        // Draw
        if (showPoint)
            circle(new_color_frame, p1[i], 2, p0[i].color, -1);
        count++;
    }
    for (int i = 0; i < p1.size(); i++)
        if (p1[i].x < old_frame.cols && p1[i].y < old_frame.rows)
            p0[i].pt = p1[i];
}

bool HumanTracker::showResult(bool stepByStep)
{
    int pauseTime = stepByStep ? 0 : 30;
    add(new_color_frame, directionMask, new_color_frame);
    add(new_color_frame, lineMask, new_color_frame);
    add(new_color_frame, mergeMask, new_color_frame);
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
        detector = GFTTDetector::create(1400, 0.03, 3, 3, false, 0.04);
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
    //double t = (double)getTickCount();

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
    putInfo("Total point amount " + std::to_string(p0.size()), 1);
    //imshow("occupied area", point_mat);

//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << t << endl;
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
//    double t = (double)getTickCount();
    if (queue_count != queue_index)
        return;
    int count = 0;
    for (int i = p0.size() - 1; i >= 0; i--) {
        if (p0[i].staticCount >= 5) {
            collectPathInfo(i);
            p0.erase(p0.begin() + i);
            count++;
        }
    }
    putInfo("Deleted static point " + std::to_string(count), 2);
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << t << endl;
}

void HumanTracker::putInfo(string text, int textY)
{
    textY *= 50;
    Rect rect(10, textY - 40, info.cols, 50);
    rectangle(info, rect, cv::Scalar(0), -1);
    cv::putText(info, text, cv::Point(10, textY), cv::FONT_HERSHEY_DUPLEX, 1.0, Scalar(255), 2);
}

void HumanTracker::addPointToPath(int queue_index)
{
//    double t = (double)getTickCount();

    if (queue_count != queue_index)
        return;

    for (int i = 0; i < p0.size(); i++)
        p0[i].updatePath();

    approximatePath();
    drawPointPath();

//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << "Time costs: " << t << endl;
}

void HumanTracker::drawPointPath()
{
    if (!showPath)
        return;

    lineMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    for (int i = 0; i < p0.size(); i++) {
        // Filter
        if (p0[i].path.size() > 2) {
            // Draw
            for (int j = 1; j < p0[i].path.size(); j++)
                line(lineMask, p0[i].path[j], p0[i].path[j - 1],
                     Scalar(0, p0[i].averageVelocity * 40, 255 - p0[i].averageVelocity * 40), 2);
        }
    }
}

void HumanTracker::approximatePath()
{
    if (showApproximanedPath)
        lineMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    if (showDirection)
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
                         ? Scalar(0, p0[i].averageVelocity * 40, 255 - p0[i].averageVelocity * 40)
                         : Scalar(255, 0, 0);
            // Draw
            if (showApproximanedPath)
                for (int j = 1; j < apx.size(); j++)
                    line(lineMask, apx[j], apx[j - 1], color, 2);
            if (p0[i].goodPath)
                drawDirection(apx, i);
        }
    }
    // Delete bad point
    int count = 0;
    for (int i = p0.size() - 1; i >= 0; i--) {
        if (!p0[i].goodPath) {
            collectPathInfo(i);
            p0.erase(p0.begin() + i);
            count++;
        }
    }
    putInfo("Deleted bad point " + std::to_string(count), 3);
}

void HumanTracker::drawDirection(vector<Point2f> &apx, int index)
{
    if (apx.size() < 2)
        return;

    Point2f pointDirection;
    int predictDiv = 2;
    float p0x = apx[apx.size() - 2].x;
    float p1x = apx[apx.size() - 1].x;
    float deltax = p1x - p0x;
    pointDirection.x = p1x + deltax / predictDiv;

    float p0y = apx[apx.size() - 2].y;
    float p1y = apx[apx.size() - 1].y;
    float deltay = p1y - p0y;
    pointDirection.y = p1y + deltay / predictDiv;
    int angle = round(atan2(- deltay, deltax) * 180 / 3.14);
    if (angle < 0)
        angle = 360 + angle;

    Scalar dirColor = cvtAngleToBGR(angle);
    p0[index].color = dirColor;
    p0[index].dirColor = true;

    if (showDirection)
        arrowedLine(directionMask, apx[apx.size() - 1], pointDirection, dirColor, 1);
}

void HumanTracker::fillHSV2BGR()
{
    for (int i = 0; i < 180; i++) {
        Mat bgr;
        Mat hsv(1,1, CV_8UC3, Scalar(i, 255, 255));
        cvtColor(hsv, bgr, COLOR_HSV2BGR);
        angleToBGR.push_back(Scalar(bgr.data[0], bgr.data[1], bgr.data[2]));
    }
}

Scalar HumanTracker::cvtAngleToBGR(int angle)
{
    if (angle > 360 || angle < 0)
        return Scalar(0, 0, 0);

    int index = angle/2 - 1;

    if (index < 0 || index > 179)
        return angleToBGR[0];
    else
        return angleToBGR[index];
}

void HumanTracker::mergePointToObject(int queue_index, int chanels)
{
//    double t = (double)getTickCount();
    if (!showMergePoint)
        return;

    if (queue_count != queue_index)
        return;

    // Draw point
    mergeMask = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    for (int i = 0; i < p0.size(); i++)
        if (p0[i].dirColor)
            circle(mergeMask, p0[i].pt, 4, p0[i].color, -1);

    Mat mergeMaskHSV = Mat::zeros(new_color_frame.size(), new_color_frame.type());
    cvtColor(mergeMask, mergeMaskHSV, COLOR_BGR2HSV);
    int angleStep = 180 / chanels;
    for (int i = 0; i < chanels; i++) {
        Mat inRangeMat = Mat::zeros(new_color_frame.size(), new_color_frame.type());
        inRange(mergeMaskHSV, Scalar(angleStep * i, 255, 255), Scalar(angleStep * (i + 1), 255, 255), inRangeMat);

        vector<vector<Point>> contours;
        cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 6));
        cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(inRangeMat, inRangeMat, rectKernel);
        cv::dilate(inRangeMat, inRangeMat, rectKernel);
        cv::erode(inRangeMat, inRangeMat, squareKernel);
        cv::erode(inRangeMat, inRangeMat, squareKernel);
        cv::findContours(inRangeMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<vector<Point> > convexHulls(contours.size());
        for (unsigned int i = 0; i < contours.size(); i++)
            convexHull(contours[i], convexHulls[i]);

        Scalar color = Scalar(angleStep * i + angleStep / 2, 255, 255);
        drawContours(mergeMaskHSV, convexHulls, -1, color, -1);
    }

    cvtColor(mergeMaskHSV, mergeMask, COLOR_HSV2BGR);
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << t << endl;
}

void HumanTracker::collectPathInfo(int index)
{
    if (!p0[index].dirColor)
        return;

    goodPathLifeTimeSum += frame_count - p0[index].originFrameCount;
    deletedGoodPathAmount++;
}

void HumanTracker::showPathInfo(int queue_index)
{
    if (queue_count != queue_index || deletedGoodPathAmount == 0)
        return;
    int averagePathLifeTime = goodPathLifeTimeSum / deletedGoodPathAmount;
    putInfo("Average path life time " + std::to_string(averagePathLifeTime), 4);
    goodPathLifeTimeSum = 0;
    deletedGoodPathAmount = 0;
}

void HumanTracker::updateMainStream(int queue_index)
{
//    double t = (double)getTickCount();
    if (queue_count != queue_index)
        return;

    for (int i = 0; i < p0.size(); ++i) {
        int x = static_cast<int>(p0[i].pt.x);
        int y = static_cast<int>(p0[i].pt.y);

        double b = mainStream.at<cv::Vec3b>(y, x)[0];
        double g = mainStream.at<cv::Vec3b>(y, x)[1];
        double r = mainStream.at<cv::Vec3b>(y, x)[2];

        if (mainStreamCount.at<int>(y, x) > 500) {
            cout << "aboba" << endl;
            mainStreamCount.at<int>(y, x) -= 100;
        }
        int count = mainStreamCount.at<int>(y, x);

        b = b * count + p0[i].color[0];
        g = g * count + p0[i].color[1];
        r = r * count + p0[i].color[2];

        count = ++mainStreamCount.at<int>(y, x);

        b /= count;
        b /= count;
        b /= count;

        mainStream.at<cv::Vec3b>(y, x)[0] = b;
        mainStream.at<cv::Vec3b>(y, x)[1] = g;
        mainStream.at<cv::Vec3b>(y, x)[2] = r;
    }
    imshow("mainStream", mainStream);
//    t = ((double)getTickCount() - t)/getTickFrequency();
    //    cout << t << endl;
}

void HumanTracker::trajectoryAnalysis(int queue_index)
{
    //double t = (double)getTickCount();

    if (queue_count != queue_index)
        return;

    double abnormalTreshold = 30;
    int initializationIterationsAmount = 8;
    int averageSelectionSize = 50;
    normalPointVelocityAmount = 0;
    abnormalPointVelocityAmount = 0;

    for (int i = 0; i < p0.size(); i++) {
        if (p0[i].averageVelocity) {
            double ratio = abs(p0[i].instantVelocity/p0[i].averageVelocity - 1) * 100;
            ratio > abnormalTreshold ? ++abnormalPointVelocityAmount : ++normalPointVelocityAmount;
        }
    }

    if (normalPointVelocityAmount != 0) {
        double currentVelocityRatio = static_cast<double>(abnormalPointVelocityAmount)
                                      /static_cast<double>(normalPointVelocityAmount);

        if (averageVelocityRatio != 0 && currentVelocityRatio/averageVelocityRatio > 2.0 && averageVelocityRatioCount > initializationIterationsAmount)
            ++abnormalOutliersFlag;
        else {
            abnormalOutliersFlag = 0;
            averageVelocityRatio *= averageVelocityRatioCount;
            averageVelocityRatio += currentVelocityRatio;
            ++averageVelocityRatioCount;
            if (averageVelocityRatioCount > averageSelectionSize) {
                averageVelocityRatio -= averageVelocityRatio / averageVelocityRatioCount;
                --averageVelocityRatioCount;
            }
            averageVelocityRatio /= averageVelocityRatioCount;
        }

        abnormalOutliersFlag > 1 ? putInfo("ABNORMAL behavior", 6) : putInfo("Normal behavior", 6);
        cout << currentVelocityRatio << "\t" << averageVelocityRatio << "\t" << averageVelocityRatioCount << endl;
    }

//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << "Time costs: " << t << endl;
}

FPoint::FPoint()
{
    pt = Point2f(0, 0);
    staticCount = 0;
    instantVelocity = 0;
    averageVelocity = 0;
    originFrameCount = 0;
    goodPath = true;
    dirColor = false;
    color = generateColor();
}

FPoint::FPoint(Point2f point, int originFrame)
{
    pt = point;
    staticCount = 0;
    instantVelocity = 0;
    averageVelocity = 0;
    originFrameCount = originFrame;
    goodPath = true;
    dirColor = false;
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
    if (path.size() < 10)
        path.push_back(pt);
    else {
        path.erase(path.begin());
        path.push_back(pt);
    }
    return updateVelocity();
}

void FPoint::updateVelocity()
{
    int instancePointAmount = 3;
    if (path.size() < instancePointAmount * 2)
        return;

    vector<Point2f> instantPath = path;
    instantPath.erase(instantPath.begin(), instantPath.end() - instancePointAmount);
    instantVelocity = cv::arcLength(instantPath, false) / (instantPath.size() - 1);

    vector<Point2f> averagePath = path;
    averagePath.erase(averagePath.end() - instancePointAmount, averagePath.end());
    averageVelocity = cv::arcLength(averagePath, false) / (averagePath.size() - 1);
}


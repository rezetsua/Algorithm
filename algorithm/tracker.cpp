#include "tracker.h"

HumanTracker::HumanTracker(const string& filename, int detector_enum)
{
    capture.open(filename);
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;

    capture >> old_frame_color;
    frame_count = 1;
    queue_count = 1;
    deletedGoodPathAmount = 0;
    goodPathLifeTimeSum = 0;
    averageVelocityRatio = 0;
    abnormalOutliersFlag = 0;
    averageVelocityRatioCount = 0;
    dataCollectionCount = 0;
    globalComm = 0;
    xPatchDim = 6;
    yPatchDim = 4;

    lineMask = Mat::zeros(old_frame_color.size(), old_frame_color.type());
    directionMask = Mat::zeros(old_frame_color.size(), old_frame_color.type());
    mergeMask = Mat::zeros(old_frame_color.size(), old_frame_color.type());
    mainStream = Mat::zeros(old_frame_color.size(), old_frame_color.type());
    patchCommMask = Mat::zeros(old_frame_color.size(), old_frame_color.type());
    gridMask = Mat::zeros(old_frame_color.size(), CV_8UC1);

    cvtColor(old_frame_color, old_frame, COLOR_BGR2GRAY);

    info = Mat::zeros(480, 480, old_frame.type());
    point_mat = Mat::zeros(old_frame.size(), old_frame.type());
    coordinateToPatchID = Mat::zeros(old_frame.size(), old_frame.type());
    mainStreamCount = Mat::zeros(old_frame.size(), CV_32S);

    fillHSV2BGR();
    fillAngleToShift();
    fillCoordinateToPatchID();
    fillGridMask();
    fillPatches();
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

        calculateOpticalFlow(LUCAS_KANADA);

        detectNewPoint(new_frame, 1);

        filterAndDrawPoint();

        deleteStaticPoint(2);

        addPointToPath(2);

        //trajectoryAnalysis(2);

        updateHOT(2);

        calcPatchHOT(2);

        calcPatchCommotion(2);

        showPatchGist(2);

        showPatchComm(2);

        mergePointToObject(3, 12);

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
    if (queue_count > 3)
        queue_count = 1;
    return true;
}

void HumanTracker::calculateOpticalFlow(int flow_enum)
{
//    double t = (double)getTickCount();
    if (flow_enum == LUCAS_KANADA) {
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        vector<Point2f> point0;
        for (int i = 0; i < p0.size(); i++)
            point0.push_back(p0[i].pt);

        calcOpticalFlowPyrLK(old_frame, new_frame, point0, p1, status, err, Size(15,15), 2, criteria, 0, 1e-4 );
        int statusCount = 0;
        for (int i = 0; i < status.size(); ++i)
            if (status[i] == 0)
                ++statusCount;
        //cout << statusCount << endl;
    }

    if (flow_enum == RLOF) {
        vector<float> err;
        vector<Point2f> point0;
        optflow::RLOFOpticalFlowParameter *rlofParam = new optflow::RLOFOpticalFlowParameter();
        rlofParam->solverType = optflow::ST_BILINEAR;
        rlofParam->supportRegionType = optflow::SR_CROSS; //SR_FIXED
        rlofParam->normSigma0 = std::numeric_limits<float>::max();
        rlofParam->normSigma1 = std::numeric_limits<float>::max();
        rlofParam->smallWinSize = 9;
        rlofParam->largeWinSize = 21;
        rlofParam->crossSegmentationThreshold = 25;
        rlofParam->maxLevel = 4;
        rlofParam->useInitialFlow = false;
        rlofParam->useIlluminationModel = true;
        rlofParam->useGlobalMotionPrior = true;
        rlofParam->maxIteration = 30;
        rlofParam->minEigenValue = 0.0001f;
        rlofParam->globalMotionRansacThreshold = 10;
        Ptr<optflow::RLOFOpticalFlowParameter> rlofParamPtr(rlofParam);
        for (int i = 0; i < p0.size(); i++)
            point0.push_back(p0[i].pt);

        optflow::calcOpticalFlowSparseRLOF(old_frame_color, new_color_frame, point0, p1, status, err, rlofParamPtr, 0);
        old_frame_color = new_color_frame.clone();
    }
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    cout << t << endl;
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
            //arrowedLine(new_color_frame, p0[i].pt, p1[i], p0[i].color, 1);
            circle(new_color_frame, p1[i], 2, p0[i].color, -1);
        count++;
    }
    for (int i = 0; i < p1.size(); i++)
        if (p1[i].x < old_frame.cols && p1[i].y < old_frame.rows
                && p1[i].x >= 0 && p1[i].y >= 0)
            p0[i].pt = p1[i];
}

bool HumanTracker::showResult(bool stepByStep)
{
    int pauseTime = stepByStep ? 0 : 30;
    add(new_color_frame, patchCommMask, new_color_frame);
    add(new_color_frame, directionMask, new_color_frame);
    add(new_color_frame, lineMask, new_color_frame);
    add(new_color_frame, mergeMask, new_color_frame);
    Mat grid = gridMask.clone();
    cvtColor(grid, grid, COLOR_GRAY2BGR);
    add(new_color_frame, grid, new_color_frame);
    //imshow("info", info);
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
    if (showApproximatedPath)
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
            if (showApproximatedPath)
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
    int predictDiv = 4;
    float p0x = apx[apx.size() - 2].x;
    float p1x = apx[apx.size() - 1].x;
    float deltax = p1x - p0x;
    pointDirection.x = p1x + deltax / predictDiv;

    float p0y = apx[apx.size() - 2].y;
    float p1y = apx[apx.size() - 1].y;
    float deltay = p1y - p0y;
    pointDirection.y = p1y + deltay / predictDiv;
    int angle = round(atan2(- deltay, deltax) * 180 / M_PI);
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

void HumanTracker::fillAngleToShift()
{
    angleToShift.resize(360);
    for (int i = 0; i < angleToShift.size(); ++i)
        for (int j = 0; j < o; ++j)
            if (i < (360 / o) * (j + 1)) {
                angleToShift[i] = j;
                break;
            }
}

void HumanTracker::fillCoordinateToPatchID()
{
    int patchWidth = coordinateToPatchID.cols / xPatchDim;
    int patchHeight = coordinateToPatchID.rows / yPatchDim;

    for (int j = 0; j < yPatchDim; ++j)
        for (int i = 0; i < xPatchDim; ++i)
            rectangle(coordinateToPatchID,
                      Rect(patchWidth * i, patchHeight * j, patchWidth, patchHeight),
                      Scalar(j * xPatchDim + i), -1);
    //imshow("coordinateToPatchID", coordinateToPatchID);
}

void HumanTracker::fillGridMask()
{
    int xStep = gridMask.cols / xPatchDim;
    int yStep = gridMask.rows / yPatchDim;
    int x = xStep;
    int y = yStep;
    while (x < gridMask.cols) {
        line(gridMask, Point2f(x, 0), Point2f(x, gridMask.rows), Scalar(255), 1);
        x += xStep;
    }
    while (y < gridMask.rows) {
        line(gridMask, Point2f(0, y), Point2f(gridMask.cols, y), Scalar(255), 1);
        y += yStep;
    }
}

void HumanTracker::fillPatches()
{
    for (int i = 0; i < xPatchDim * yPatchDim; ++i) {
        int xIndex = i % xPatchDim;
        int yIndex = i / xPatchDim;

        int x = old_frame_color.cols / (2 * xPatchDim) + xIndex * old_frame_color.cols / xPatchDim;
        int y = old_frame_color.rows / (2 * yPatchDim) + yIndex * old_frame_color.rows / yPatchDim;

        patches.push_back(Patch(Point2f(x, y)));
        circle(gridMask, patches[i].center, 2, Scalar(0, 255, 0), -1);
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

    if (!trajectoryAnalys)
        return;
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
        //cout << currentVelocityRatio << "\t" << averageVelocityRatio << "\t" << averageVelocityRatioCount << endl;
    }

//    t = ((double)getTickCount() - t)/getTickFrequency();
    //    cout << "Time costs: " << t << endl;
}

void HumanTracker::updateHOT(int queue_index)
{
    if (queue_count != queue_index)
        return;

    uint32_t hstg = 0;
    double magnStep = magnMax / m;
    for (int i = 0; i < p0.size(); ++i) {
        if (p0[i].path.size() < 2)
            return;

        Point2f pt1 = p0[i].path[p0[i].path.size() - 2];
        Point2f pt2 = p0[i].path[p0[i].path.size() - 1];

        double magn = norm(pt2 - pt1);
        int magnShift = m - 1;
        for (int i = 0; i < m; ++i)
            if (magn < (magnStep * (i + 1))) {
                magnShift = i;
                break;
            }

        int angle = round(atan2(- (pt2.y - pt1.y), pt2.x - pt1.x) * 180 / M_PI);
        if (angle < 0) angle = 360 + angle;
        int angleShift = angleToShift[angle];

        hstg = o * magnShift + angleShift;
        //hstg = 1 << (o * magnShift + angleShift);

        p0[i].newHotCount++;
        if (p0[i].newHotCount > TL)
            p0[i].newHotCount = 1;

        if (p0[i].path.size() < TL)
            p0[i].hot.push_back(hstg);
        else {
            p0[i].hot.erase(p0[i].hot.begin());
            p0[i].hot.push_back(hstg);
        }
        //cout << "magn = " << magn << " angle = " << angle << " magnShift = " << magnShift << " angleShift = " << angleShift << " hist = " << hstg << endl;
    }
}

void HumanTracker::calcPatchHOT(int queue_index)
{
    if (queue_count != queue_index)
        return;

    for (int i = 0; i < p0.size(); ++i) {
        if (p0[i].newHotCount != TL)
            continue;

        int cX = p0[i].path[0].x + (p0[i].path[TL - 1].x - p0[i].path[0].x)/2;
        int cY = p0[i].path[0].y + (p0[i].path[TL - 1].y - p0[i].path[0].y)/2;

        int patchID = coordinateToPatchID.at<uchar>(cY, cX);

        for (int j = 0; j < p0[i].hot.size(); ++j) {
            patches.at(patchID).lbt.at(o * m * j + p0[i].hot[j])++;
            patches.at(patchID).lbtLifeTime.at(o * m * j + p0[i].hot[j]) = 0;
        }
    }
}

void HumanTracker::calcPatchCommotion(int queue_index)
{
    if (queue_count != queue_index)
        return;

    dataCollectionCount++;
    cout << dataCollectionCount << "\t";
    cout << fixed;
    cout.precision(2);
//    if (dataCollectionCount < 50)
//        return;


    double commSum = 0;
    for (int i = 0; i < patches.size(); ++i) {
        patches[i].updateComm();
        commSum += patches[i].comm;
        cout << patches[i].comm << " ";
    }
    cout << "\t" << commSum << "\t" << commSum - globalComm << endl;
    globalComm = commSum;
}

void HumanTracker::showPatchGist(int queue_index)
{
    if (queue_count != queue_index)
        return;

    bool L2Mode = false;

    Mat patchGist = Mat::zeros(gridMask.size(), gridMask.type());
    add(patchGist, gridMask, patchGist);

    for (int i = 0; i < patches.size(); ++i) {

        int jmax = -1;
        double L2 = 0;
        for (int j = 0; j < patches[i].lbt.size(); ++j) {
            L2 += pow(patches[i].lbt[j], 2);
            if (patches[i].lbt[j] > patches[i].lbt[jmax])
                jmax = j;
        }
        L2 = sqrt(L2);

        double xLength = new_frame.cols / (xPatchDim * 2);
        double yLength = new_frame.rows / (yPatchDim * 2);
        double scaleM = fmin(xLength, yLength) / magnMax;

        for (int j = 0; j < patches[i].lbt.size(); ++j) {
            if (patches[i].lbt[j] == 0)
                continue;

            double oj = patches[i].indexToAngle[patches[i].getLocalIndex(j).second];
            double mj = scaleM * patches[i].indexToMagnitude[patches[i].getLocalIndex(j).first];
            double I = 255.0 * static_cast<double>(patches[i].lbt[j])
                     / (L2Mode ? L2 : patches[i].lbt[jmax]);

            int x = mj * cos(oj);
            int y = - mj * sin(oj);
            Point2f pt(patches[i].center.x + x, patches[i].center.y + y);

            line(patchGist, patches[i].center, pt, Scalar(I), 2);
        }

        if (jmax != -1) {
            double ojmax = patches[i].indexToAngle[patches[i].getLocalIndex(jmax).second];
            double mjmax = scaleM * patches[i].indexToMagnitude[patches[i].getLocalIndex(jmax).first];
            double I = L2Mode ? 255.0 / L2 : 255;

            int x = mjmax * cos(ojmax);
            int y = - mjmax * sin(ojmax);
            Point2f pt(patches[i].center.x + x, patches[i].center.y + y);

            line(patchGist, patches[i].center, pt, Scalar(I), 2);
        }
    }

    imshow("patchGist", patchGist);
}

void HumanTracker::showPatchComm(int queue_index)
{
    if (queue_count != queue_index)
        return;

    int patchWidth = patchCommMask.cols / xPatchDim;
    int patchHeight = patchCommMask.rows / yPatchDim;

    patchCommMask = Mat::zeros(patchCommMask.size(), patchCommMask.type());
    for (int j = 0; j < yPatchDim; ++j)
        for (int i = 0; i < xPatchDim; ++i) {
            double maxComm = 0.25;
            double r = 255.0 * patches[j * xPatchDim + i].comm / maxComm;
            rectangle(patchCommMask,
                      Rect(patchWidth * i, patchHeight * j, patchWidth, patchHeight),
                      Scalar(255 - r, 0, r), -1);
        }
    Mat patchCommMaskShow = patchCommMask.clone();
    Mat grid = gridMask.clone();
    cvtColor(grid, grid, COLOR_GRAY2BGR);
    add(patchCommMaskShow, grid, patchCommMaskShow);
    imshow("patchCommMaskShow", patchCommMaskShow);
}

FPoint::FPoint()
{
    pt = Point2f(0, 0);
    staticCount = 0;
    instantVelocity = 0;
    averageVelocity = 0;
    averageVelocityCount = 0;
    originFrameCount = 0;
    newHotCount = 0;
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
    averageVelocityCount = 0;
    originFrameCount = originFrame;
    newHotCount = 0;
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
    if (path.size() < TL)
        path.push_back(pt);
    else {
        path.erase(path.begin());
        path.push_back(pt);
    }
    updateVelocity();
}

void FPoint::updateVelocity()
{
    int instancePointAmount = 3;
    if (path.size() < instancePointAmount + 2)
        return;

    vector<Point2f> instantPath = path;
    instantPath.erase(instantPath.begin(), instantPath.end() - instancePointAmount);
    instantVelocity = cv::arcLength(instantPath, false) / (instantPath.size() - 1);

    averageVelocity *= averageVelocityCount;
    averageVelocity += cv::norm(path[path.size() - instancePointAmount - 1] - path[path.size() - instancePointAmount - 2]);
    ++averageVelocityCount;
    averageVelocity /= averageVelocityCount;
}

Patch::Patch(Point2f pt)
{
    center = pt;
    comm = 0;
    lbt.resize(o * m * TL);
    lbtLifeTime.resize(o * m * TL);

    indexToAngle.resize(o);
    for (int i = 0; i < indexToAngle.size(); ++i)
        indexToAngle[i] = (2 * M_PI / o) * i + M_PI / o;

    indexToMagnitude.resize(m);
    for (int i = 0; i < indexToMagnitude.size(); ++i)
        indexToMagnitude[i] = (magnMax / m) * i + magnMax / (2.0 * m);
}

void Patch::updateComm()
{
    double L2 = 0;
    int jmax = 0;
    for (int i = 0; i < lbt.size(); ++i) {
        L2 += pow(lbt[i], 2);

        if (lbt[i] > lbt[jmax])
            jmax = i;
    }
    L2 = sqrt(L2);

    comm = 0;
    for (int j = 0; j < lbt.size(); ++j) {
        if (lbt[j] != 0) {
            if (lbtLifeTime[j] > 10)
                if (lbt[j] > 0)
                    lbt[j]--;
            comm += getIndexWeight(j, jmax) * pow((lbt[j] - lbt[jmax]) / L2, 2);
            lbtLifeTime[j]++;
        }
    }
}

double Patch::getIndexWeight(int j, int jmax)
{
    double sigO = 2 * M_PI / 12.0; //1.0 / o;
    double sigM = magnMax / 12.0; //1.0 / m;

    double oj = indexToAngle[getLocalIndex(j).second];
    double ojmax = indexToAngle[getLocalIndex(jmax).second];

    double mj = indexToMagnitude[getLocalIndex(j).first];
    double mjmax = indexToMagnitude[getLocalIndex(jmax).first];

    double dMmax = mjmax > magnMax / 2 ? mjmax : magnMax - mjmax;
    double dM = std::abs(mj - mjmax);
    assert(dMmax >= dM);

    double dOmax = M_PI;
    double dO = std::abs(oj - ojmax);
    dO = dO > M_PI ? 2 * M_PI - dO : dO;
    assert(dOmax >= dO);

    double A = 1; // 1 / (2 * M_PI * sigO * sigM);
    double B = (pow(dO - dOmax, 2)) / (2 * pow(sigO, 2));
    double C = (pow(dM - dMmax, 2)) / (2 * pow(sigM, 2));

    double weight = A * exp(-1 * B - C);
    return weight;
}

std::pair<int, int> Patch::getLocalIndex(int i)
{
    int localHotIndex = i % (o * m);

    int localMagnIndex = localHotIndex / o;
    int localOrientIndex = localHotIndex % o;

    return make_pair(localMagnIndex, localOrientIndex);
}

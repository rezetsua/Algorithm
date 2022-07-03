#include "tracker.h"

HumanTracker::HumanTracker(const string& filename, int flow,  int detector, int captureMode)
{
    this->captureMode = captureMode;
    flowType = flow;
    setDetector(detector);
    frameCount = 1;
    queueCount = 1;
    deletedGoodPathAmount = 0;
    goodPathLifeTimeSum = 0;

    if (captureMode == VIDEO_CAPTURE) {
        capture.open(filename);
        fillGroundTruthTXT(filename);
    }
    else if (captureMode == IMAGE_CAPTURE) {
        size_t found = filename.find_last_of(".");
        string format = filename.substr(found);
        found = filename.find_last_of("/");
        string cap = filename.substr(0, found) + "/%03d" + format;
        capture.open(cap);
        fillGroundTruthIMG(filename);
    }
    if (!capture.isOpened())
        cerr << "Unable to open file!" << endl;
    capture >> oldFrameColor;

    lineMask = Mat::zeros(oldFrameColor.size(), oldFrameColor.type());
    directionMask = Mat::zeros(oldFrameColor.size(), oldFrameColor.type());
    mergeMask = Mat::zeros(oldFrameColor.size(), oldFrameColor.type());
    patchCommMask = Mat::zeros(oldFrameColor.size(), oldFrameColor.type());
    gridMask = Mat::zeros(oldFrameColor.size(), CV_8UC1);
    cvtColor(oldFrameColor, oldFrame, COLOR_BGR2GRAY);
    info = Mat::zeros(480, 480, oldFrame.type());
    pointMat = Mat::zeros(oldFrame.size(), oldFrame.type());
    coordinateToPatchID = Mat::zeros(oldFrame.size(), oldFrame.type());

    fillHSV2BGR();
    fillAngleToShift();
    fillCoordinateToPatchID();
    fillGridMask();
    fillPatches();

    // Makes first detection for flow algorithm
    detectNewPoint(oldFrame, 1);
}

void HumanTracker::stopTracking()
{
    running = false;
}

void HumanTracker::startTracking()
{
    running = true;

    double t;
    while(running){
        t = (double)getTickCount();

        if (!getNextFrame()) break;

        calculateOpticalFlow(flowType);

        detectNewPoint(newFrame, 1);

        filterAndDrawPoint();

        deleteStaticPoint(2);

        addPointToPath(2);

        trajectoryAnalysis(2);

        //mergePointToObject(3, 12);

        double dt = ((double)getTickCount() - t)/getTickFrequency();
        computingTimeCost += dt;
        if (frameCount % 10 == 0)
            putInfo("FPS " + std::to_string((int)(1/dt)), 5);

        if (!showResult(false)) break;
    }

    printInfo();
    waitKey(0);
}

bool HumanTracker::getNextFrame()
{
    capture >> newColorFrame;
    if (newColorFrame.empty())
        return false;

    cvtColor(newColorFrame, newFrame, COLOR_BGR2GRAY);

    frameCount++;
    queueCount++;
    if (queueCount > queueIteration)
        queueCount = 1;

    return true;
}

void HumanTracker::calculateOpticalFlow(int flow_enum)
{
    if (flow_enum == LUCAS_KANADA) {
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

        // Get Point2f from custom FPoint array
        vector<Point2f> point0;
        for (int i = 0; i < p0.size(); i++)
            point0.push_back(p0[i].pt);

        calcOpticalFlowPyrLK(oldFrame, newFrame, point0, p1, status, err, Size(21,21), 2, criteria, 0, 0.1);
    }

    if (flow_enum == RLOF) {
        vector<float> err;
        optflow::RLOFOpticalFlowParameter *rlofParam = new optflow::RLOFOpticalFlowParameter();
        rlofParam->solverType = optflow::ST_STANDART;
        rlofParam->supportRegionType = optflow::SR_FIXED;
        rlofParam->normSigma0 = std::numeric_limits<float>::max();
        rlofParam->normSigma1 = std::numeric_limits<float>::max();
        rlofParam->smallWinSize = 9;
        rlofParam->largeWinSize = 21;
        rlofParam->crossSegmentationThreshold = 30; // Important if supportRegionType = SR_CROSS
        rlofParam->maxLevel = 3;
        rlofParam->useInitialFlow = false;
        rlofParam->useIlluminationModel = false;
        rlofParam->useGlobalMotionPrior = false;
        rlofParam->maxIteration = 20;
        rlofParam->minEigenValue = 1e-3;
        rlofParam->globalMotionRansacThreshold = 20;
        Ptr<optflow::RLOFOpticalFlowParameter> rlofParamPtr(rlofParam);

        // Get Point2f from custom FPoint array
        vector<Point2f> point0;
        for (int i = 0; i < p0.size(); i++)
            point0.push_back(p0[i].pt);

        optflow::calcOpticalFlowSparseRLOF(oldFrameColor, newColorFrame, point0, p1, status, err, rlofParamPtr, 0);
        oldFrameColor = newColorFrame.clone();
    }
}

void HumanTracker::filterAndDrawPoint()
{
    for(int i = 0; i < p1.size(); i++)
    {
        // Mark static point
        double delta = cv::norm(p1[i] - p0[i].pt);
        float treshold = 0.3;
        if (delta < treshold ) {
            p0[i].staticCount++;
            continue;
        }
        p0[i].staticCount = 0;

        // Draw point
        if (showPoint)
            circle(newColorFrame, p1[i], 2, p0[i].color, -1);
    }

    // Fill FPoint array by new point's coordinate
    for (int i = 0; i < p1.size(); i++)
        if (p1[i].x < oldFrame.cols && p1[i].y < oldFrame.rows && p1[i].x >= 0 && p1[i].y >= 0)
            p0[i].pt = p1[i];
}

bool HumanTracker::showResult(bool stepByStep)
{
    add(newColorFrame, directionMask, newColorFrame);
    add(newColorFrame, lineMask, newColorFrame);
    add(newColorFrame, mergeMask, newColorFrame);
    Mat grid = gridMask.clone();
    cvtColor(grid, grid, COLOR_GRAY2BGR);
    add(newColorFrame, grid, newColorFrame);
    imshow("flow", newColorFrame);

    Mat analysis = newColorFrame.clone();
    add(analysis, patchCommMask, analysis);
    addAnomalyTitle(analysis);
    imshow("analysis", analysis);

    imshow("info", info);

    oldFrame = newFrame.clone();

    int pauseTime = stepByStep ? 0 : waitkeyPause;
    if (waitKey(pauseTime) == 27)
        return stepByStep;
    return true;
}

void HumanTracker::setDetector(int detector_enum)
{
    switch (detector_enum) {
    case GFTT_Detector: {
        detector = GFTTDetector::create(500, 0.03, 3, 3, false, 0.03);
        break;
    }
    case FAST_Detector: {
        detector = FastFeatureDetector::create(80, true, FastFeatureDetector::TYPE_9_16);
        break;
    }
    case AGAST_Detector: {
        detector = AgastFeatureDetector::create(40, false, AgastFeatureDetector::AGAST_7_12s);
        break;
    }
    case SimpleBlob_Detector: {
        SimpleBlobDetector::Params params;
        // Change thresholds
        params.minThreshold = 20;
        params.maxThreshold = 100;
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
    case SIFT_Detector: {
        detector = SIFT::create(0, 5, 0.08, 30, 1.0);
        break;
    }
    case MSER_Detector: {
        detector = MSER::create(1, 10, 60, 0.5);
        break;
    }
    case KAZE_Detector: {
        detector = KAZE::create(false, false, 0.001f, 4, 5, KAZE::DIFF_CHARBONNIER);
        break;
    }
    case AKAZE_Detector: {
        detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 100, 3, 0.0005f, 4, 2, KAZE::DIFF_CHARBONNIER);
        break;
    }
    }
}

void HumanTracker::detectNewPoint(Mat &frame, int queue_index)
{
    if (queueCount != queue_index)
        return;

    detector->detect(frame, newPoint);

    // Adds only really new points
    int oldPointArea = 7;
    fillPointMat(oldPointArea);
    for (int i = 0; i < newPoint.size(); i++) {
        if (pointMat.at<uchar>(newPoint[i].pt.y, newPoint[i].pt.x) == 255)
            continue;
        p0.push_back(FPoint(newPoint[i].pt, frameCount));
        circle(pointMat, newPoint[i].pt, oldPointArea, Scalar(255), -1);
    }

    putInfo("Total point amount " + std::to_string(p0.size()), 1);
}

void HumanTracker::fillPointMat(int blockSize)
{
    pointMat = Mat::zeros(oldFrame.size(), oldFrame.type());
    for (int i = 0; i < p0.size(); i++) {
        circle(pointMat, p0[i].pt, blockSize, Scalar(255), -1);
    }
}

void HumanTracker::deleteStaticPoint(int queue_index)
{
    if (queueCount != queue_index)
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
}

void HumanTracker::putInfo(string text, int textY)
{
    textY *= 50; // Order of labels from top to bottom
    Rect rect(10, textY - 40, info.cols, 50);
    rectangle(info, rect, cv::Scalar(0), -1);
    cv::putText(info, text, cv::Point(10, textY), cv::FONT_HERSHEY_DUPLEX, 1.0, Scalar(255), 2);
}

void HumanTracker::addPointToPath(int queue_index)
{
    if (queueCount != queue_index)
        return;

    for (int i = 0; i < p0.size(); i++)
        p0[i].updatePath();

    approximatePath();
    drawPointPath();
    showPathInfo(2);
}

void HumanTracker::drawPointPath()
{
    if (!showPath)
        return;

    lineMask = Mat::zeros(newColorFrame.size(), newColorFrame.type());
    for (int i = 0; i < p0.size(); i++) {
        // Filter
        if (p0[i].path.size() < 3)
            continue;

        // Draw
        for (int j = 1; j < p0[i].path.size(); j++)
            line(lineMask, p0[i].path[j], p0[i].path[j - 1],
                 Scalar(0, p0[i].averageVelocity * 255 / magnMax, 255 - p0[i].averageVelocity * 255 / magnMax), 2);
    }
}

void HumanTracker::approximatePath()
{
    if (showApproximatedPath)
        lineMask = Mat::zeros(newColorFrame.size(), newColorFrame.type());

    if (showDirection)
        directionMask = Mat::zeros(newColorFrame.size(), newColorFrame.type());

    vector<Point2f> apx;
    for (int i = 0; i < p0.size(); i++) {
        if (p0[i].path.size() > 2) {
            // Approximate
            double epsilon = 0.2 * arcLength(p0[i].path, false);
            approxPolyDP(p0[i].path, apx, epsilon, false);

            // Filter (if the trajectory is approximated in a straight line, occlusion has not occurred)
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
    usfullPointAmount += p0.size();
    usfullPointCount++;
}

void HumanTracker::drawDirection(vector<Point2f> &apx, int index)
{
    if (apx.size() < 2)
        return;

    Point2f pointDirection;
    double predictDiv = 1.5; // smaller value -> longer vector in the image

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

    // Color coding of the direction
    Scalar dirColor = cvtAngleToBGR(angle);
    p0[index].color = dirColor;
    p0[index].dirColor = true;

    // Draw
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

        int x = oldFrameColor.cols / (2 * xPatchDim) + xIndex * oldFrameColor.cols / xPatchDim;
        int y = oldFrameColor.rows / (2 * yPatchDim) + yIndex * oldFrameColor.rows / yPatchDim;

        patches.push_back(Patch(Point2f(x, y)));
    }
}

void HumanTracker::fillGroundTruthTXT(string filename)
{
    size_t found = filename.find_last_of(".");
    filename.erase(found + 1, filename.size());
    filename.append("txt");

    ifstream f(filename);
    stringstream ss;
    ss << f.rdbuf();
    string str = ss.str();

    groundTruth.resize(str.size());
    for (int i = 0; i < str.size(); ++i)
        groundTruth[i] = static_cast<int>(str[i]) - 48;
}

void HumanTracker::fillGroundTruthIMG(string filename)
{
    size_t found = filename.find_last_of("/");
    string cap = filename.substr(0, found) + "_gt/%03d.bmp";
    VideoCapture captureGT(cap);
    Mat gt;

    while (true) {
        captureGT >> gt;
        if (gt.empty())
            break;

        cvtColor(gt, gt, COLOR_BGR2GRAY);
        vector<int> gtV;
        gtV.resize(xPatchDim * yPatchDim);
        for (int j = 0; j < gt.rows; ++j)
            for (int i = 0; i < gt.cols; ++i)
                if (gt.at<uchar>(j, i) > 200)
                    gtV[coordinateToPatchID.at<uchar>(j, i)] = 1;

        for (int i = 0; i < gtV.size(); ++i)
            groundTruth.push_back(gtV[i]);
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
    if (!showMergePoint)
        return;

    if (queueCount != queue_index)
        return;

    // Draw point
    mergeMask = Mat::zeros(newColorFrame.size(), newColorFrame.type());
    for (int i = 0; i < p0.size(); i++)
        if (p0[i].dirColor)
            circle(mergeMask, p0[i].pt, 4, p0[i].color, -1);

    Mat mergeMaskHSV = Mat::zeros(newColorFrame.size(), newColorFrame.type());
    cvtColor(mergeMask, mergeMaskHSV, COLOR_BGR2HSV);
    int angleStep = 180 / chanels;
    for (int i = 0; i < chanels; i++) {
        // Сolor segmentation
        Mat inRangeMat = Mat::zeros(newColorFrame.size(), newColorFrame.type());
        inRange(mergeMaskHSV, Scalar(angleStep * i, 255, 255), Scalar(angleStep * (i + 1), 255, 255), inRangeMat);

        // Сombining the nearest contours
        vector<vector<Point>> contours;
        cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 6));
        cv::Mat squareKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(inRangeMat, inRangeMat, rectKernel);
        cv::dilate(inRangeMat, inRangeMat, rectKernel);
        cv::erode(inRangeMat, inRangeMat, squareKernel);
        cv::erode(inRangeMat, inRangeMat, squareKernel);
        cv::findContours(inRangeMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Get convex hulls
        vector<vector<Point> > convexHulls(contours.size());
        for (unsigned int i = 0; i < contours.size(); i++)
            convexHull(contours[i], convexHulls[i]);

        // Drawing the hulls with the average color of the sector
        Scalar color = Scalar(angleStep * i + angleStep / 2, 255, 255);
        drawContours(mergeMaskHSV, convexHulls, -1, color, -1);
    }

    cvtColor(mergeMaskHSV, mergeMask, COLOR_HSV2BGR);
}

void HumanTracker::collectPathInfo(int index)
{
    if (!p0[index].dirColor)
        return;

    goodPathLifeTimeSum += frameCount - p0[index].originFrameCount;
    deletedGoodPathAmount++;
}

void HumanTracker::showPathInfo(int queue_index)
{
    if (queueCount != queue_index || deletedGoodPathAmount == 0)
        return;
    int averagePathLifeTime = goodPathLifeTimeSum / deletedGoodPathAmount;
    putInfo("Average path life time " + std::to_string(averagePathLifeTime), 4);
}

void HumanTracker::trajectoryAnalysis(int queue_index)
{
    if (queueCount != queue_index)
        return;

    updateHOT(queue_index);
    calcPatchHOT(queue_index);
    calcPatchCommotion(queue_index);
    showPatchGist(queue_index);
    showPatchComm(queue_index);
}

void HumanTracker::updateHOT(int queue_index)
{
    if (queueCount != queue_index)
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

        // Instead of an array of o*m elements, the position of 1 in the array is stored
        hstg = o * magnShift + angleShift;

        p0[i].newHotCount++;
        if (p0[i].newHotCount > TL)
            p0[i].newHotCount = 1;

        // Store only the latest TL hstg
        if (p0[i].path.size() < TL)
            p0[i].hot.push_back(hstg);
        else {
            p0[i].hot.erase(p0[i].hot.begin());
            p0[i].hot.push_back(hstg);
        }
    }
}

void HumanTracker::calcPatchHOT(int queue_index)
{
    if (queueCount != queue_index)
        return;

    for (int i = 0; i < p0.size(); ++i) {
        // Counts HOT only on new data
        if (p0[i].newHotCount != TL)
            continue;

        // Coordinates of the center of the tracklet
        int cX = p0[i].path[0].x + (p0[i].path[TL - 1].x - p0[i].path[0].x) / 2.0;
        int cY = p0[i].path[0].y + (p0[i].path[TL - 1].y - p0[i].path[0].y) / 2.0;

        assert((cX < oldFrame.cols && cY < oldFrame.rows && cX >= 0 && cY >= 0));

        int patchID = coordinateToPatchID.at<uchar>(cY, cX);
        if (patches.at(patchID).isEmpty)
            patchInit(patchID);

        // Add new data in HOT
        for (int j = 0; j < p0[i].hot.size(); ++j) {
            patches.at(patchID).lbt.at(o * m * j + p0[i].hot[j])++;
            patches.at(patchID).lbtLifeTime.at(o * m * j + p0[i].hot[j]) = 0;
        }
    }
}

void HumanTracker::calcPatchCommotion(int queue_index)
{
    if (queueCount != queue_index)
        return;

    double commSum = 0;
    for (int i = 0; i < patches.size(); ++i) {
        patches[i].updateComm();
        commSum += patches[i].comm;

        // Filling the probability of anomaly and groundtruth for the current patch of the frame
        if (anomalyType == LOCALIZATION) {
            prob.push_back(patches[i].comm);
            truth.push_back(groundTruth[(frameCount - 1) * xPatchDim * yPatchDim + i]);
        }
    }
    // Filling the probability of anomaly and groundtruth for the current frame
    if (anomalyType == DETECTION) {
        prob.push_back(commSum);
        truth.push_back(groundTruth[frameCount - 1]);
    }

    commSum > commFrameTresh ? anomaly = true : anomaly = false;
}

void HumanTracker::showPatchGist(int queue_index)
{
    if (queueCount != queue_index)
        return;

    bool L2Mode = false;
    Mat patchGist = Mat::zeros(oldFrameColor.size(), oldFrameColor.type());
    Mat grid = gridMask.clone();
    cvtColor(grid, grid, COLOR_GRAY2BGR);
    add(patchGist, grid, patchGist);

    for (int i = 0; i < patches.size(); ++i) {
        // Calculates index of max value in patches[i].lbt and L2 for patches[i].lbt
        int jmax = -1;
        double L2 = 0;
        for (int j = 0; j < patches[i].lbt.size(); ++j) {
            L2 += pow(patches[i].lbt[j], 2);
            if (patches[i].lbt[j] > patches[i].lbt[jmax])
                jmax = j;
        }
        L2 = sqrt(L2);

        // Calculates the coefficient(scaleM) for the correct display of the displacement value
        double xLength = newFrame.cols / (xPatchDim * 2);
        double yLength = newFrame.rows / (yPatchDim * 2);
        double scaleM = fmin(xLength, yLength) / magnMax;

        // Draws each movement of patch[i].lbt from the center of the patch, preserving the angle and magnitude
        // More green color -> more often this movement occurs in patch[i].lbt
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

            line(patchGist, patches[i].center, pt, Scalar(0, I/2, (255 - I) / 2), 2);
        }

        // Re-draws the most popular direction on top of others for a better perception
        if (jmax != -1) {
            double ojmax = patches[i].indexToAngle[patches[i].getLocalIndex(jmax).second];
            double mjmax = scaleM * patches[i].indexToMagnitude[patches[i].getLocalIndex(jmax).first];
            double I = L2Mode ? 255.0 / L2 : 255;

            int x = mjmax * cos(ojmax);
            int y = - mjmax * sin(ojmax);
            Point2f pt(patches[i].center.x + x, patches[i].center.y + y);

            line(patchGist, patches[i].center, pt, Scalar(0, I, (255 - I) / 2), 2);
        }
    }

    imshow("patchGist", patchGist);
}

void HumanTracker::showPatchComm(int queue_index)
{
    if (queueCount != queue_index)
        return;

    int patchWidth = patchCommMask.cols / xPatchDim;
    int patchHeight = patchCommMask.rows / yPatchDim;

    patchCommMask = Mat::zeros(patchCommMask.size(), patchCommMask.type());
    for (int j = 0; j < yPatchDim; ++j)
        for (int i = 0; i < xPatchDim; ++i) {
            // More red(r) -> higher probability of an abnormal event in the patch
            double r = 255.0 * patches[j * xPatchDim + i].comm / commPatchTresh;
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

void HumanTracker::patchInit(int index)
{
    if (!predictPatchLBT)
        return;

    // Finds all neighbors of the patch depending on the mode (bigPatchInit)
    vector<int> neighbors;
    bool left = index % xPatchDim != 0;
    bool right = (index + 1) % xPatchDim != 0;
    bool top = index - xPatchDim >= 0;
    bool down = index + xPatchDim < xPatchDim * yPatchDim;

    if (left)
        neighbors.push_back(index - 1);
    if (right)
        neighbors.push_back(index + 1);
    if (top)
        neighbors.push_back(index - xPatchDim);
    if (down)
        neighbors.push_back(index + xPatchDim);

    if (bigPatchInit) {
        if (left && top)
            neighbors.push_back(index - xPatchDim - 1);
        if (right && top)
            neighbors.push_back(index - xPatchDim + 1);
        if (left && down)
            neighbors.push_back(index + xPatchDim - 1);
        if (right && down)
            neighbors.push_back(index + xPatchDim + 1);
    }

    // Add the most popular moves from neighboring patches to the patch
    // The popularity of moves can be resize by the patchInitWeight
    for (int i = 0; i < neighbors.size(); i++) {
        if (patches.at(neighbors[i]).isEmpty)
            continue;

        int jmax = 0;
        for (int j = 0; j < patches.at(neighbors[i]).lbt.size(); ++j) {
            if (patches.at(neighbors[i]).lbt[j] > patches.at(neighbors[i]).lbt[jmax])
                jmax = j;
        }
        patches.at(index).lbt[jmax] = patchInitWeight * patches.at(neighbors[i]).lbt[jmax];
        patches.at(index).lbtLifeTime[jmax] = 0;
    }
    patches.at(index).isEmpty = false;
}

void HumanTracker::printInfo()
{
    int avgPathLifeTime = goodPathLifeTimeSum / deletedGoodPathAmount;
    int avgPointAmount = usfullPointAmount / usfullPointCount;
    int avgFPS = frameCount / computingTimeCost;

    cout << "Average usfull point amount = " << avgPointAmount << endl;
    cout << "Average path life time = " << avgPathLifeTime << endl;
    cout << "Average FPS = " << avgFPS << endl;
}

void HumanTracker::addAnomalyTitle(Mat &img)
{
    Rect rect(10, 10, 180, 30);
    Scalar color = anomaly ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
    string text = anomaly ? "Abnormal behavior" : "Normal behavior";
    rectangle(img, rect, color, -1);
    cv::putText(img, text, cv::Point(30, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, Scalar(0), 2);
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
    // Store only the latest TL points
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
    int instancePointAmount = 3; // The number of points to calculate the instantaneous speed
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
    isEmpty = true;
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
    // Calculates index of max value in lbt and L2 for lbt
    double L2 = 0;
    int jmax = 0;
    for (int i = 0; i < lbt.size(); ++i) {
        L2 += pow(lbt[i], 2);

        if (lbt[i] > lbt[jmax])
            jmax = i;
    }
    L2 = sqrt(L2);
    isEmpty = (L2 < 1);

    // Calculates commotion
    comm = 0;
    for (int j = 0; j < lbt.size(); ++j) {
        if (lbt[j] == 0)
            continue;
        if (lbtLifeTime[j] > lbtLifeTimeDelta)
            lbtResetLifeTime ? lbt[j] = 0 : lbt[j]--;
        double currComm = getIndexWeight(j, jmax) * pow((lbt[j] - lbt[jmax]) / L2, 2);
        comm += currComm;
        lbtLifeTime[j]++;
    }
}

double Patch::getIndexWeight(int j, int jmax)
{
    double oj = indexToAngle[getLocalIndex(j).second];
    double ojmax = indexToAngle[getLocalIndex(jmax).second];

    double mj = indexToMagnitude[getLocalIndex(j).first];
    double mjmax = indexToMagnitude[getLocalIndex(jmax).first];

    // If the abnormal behavior is the most different from the average
//    double dMmax = mjmax > magnMax / 2 ? mjmax : magnMax - mjmax;
//    double dM = std::abs(mj - mjmax);

    // If the abnormal behavior is faster than average
    double dMmax = magnMax - mjmax;
    double dM = mj - mjmax;

    double dOmax = M_PI;
    double dO = std::abs(oj - ojmax);
    dO = dO > M_PI ? 2 * M_PI - dO : dO;

    double sigO = dOmax / 6.0;
    double sigM = dMmax / 6.0;

    double A = 1;
    double B = (pow(dO - dOmax, 2)) / (2 * pow(sigO, 2));
    double C = (pow(dM - dMmax, 2)) / (2 * pow(sigM, 2));

    double weight = 0;
    switch (anomalyCalcMode) {
    case MAGNITUDE:
        weight = A * exp(-1 * C);
        break;
    case DIRECTION:
        weight = A * exp(-1 * B);
        break;
    case BOTH:
        weight = A * exp(-1 * B - C);
        break;
    default:
        weight = A * exp(-1 * B - C);
        break;
    }
    return weight;
}

std::pair<int, int> Patch::getLocalIndex(int i)
{
    int localHotIndex = i % (o * m);

    int localMagnIndex = localHotIndex / o;
    int localOrientIndex = localHotIndex % o;

    return make_pair(localMagnIndex, localOrientIndex);
}

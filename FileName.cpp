#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
cv::Mat circleDetect(cv::Mat frame) {
    cv::Mat out = frame;
    ////中值滤波
    cv::medianBlur(frame, out, 1);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
    //边缘检测
    cv::equalizeHist(out, out);
    cv::Mat edges_image;
    Canny(out, edges_image, 100, 300);
    // 形态学操作 - 闭运算
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges_image, edges_image, cv::MORPH_CLOSE, kernel);
    //提取轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    //找椭圆
    cv::RotatedRect largestEllipse;
    double largestEllipseArea = 0.0;
    for (int i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if (count < 100 || count > 500) continue;
        cv::Mat pointsf;
        cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
        cv::RotatedRect box = fitEllipseDirect(pointsf);
        double currentEllipseArea = box.size.width * box.size.height;
        if (currentEllipseArea > largestEllipseArea)
        {
            largestEllipse = box;
            largestEllipseArea = currentEllipseArea;
        }
    }
    if (largestEllipseArea > 0.0)
    {
        ellipse(frame, largestEllipse, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::Point center = largestEllipse.center;
        double longAxis = largestEllipse.size.height > largestEllipse.size.width ?
            largestEllipse.size.height : largestEllipse.size.width;
        double shortAxis = largestEllipse.size.height > largestEllipse.size.width ?
            largestEllipse.size.width : largestEllipse.size.height;
        double angle = largestEllipse.angle;

        std::cout << "椭圆参数：" << std::endl;
        std::cout << "圆心坐标：(" << center.x << ", " << center.y << ")" << std::endl;
        std::cout << "长半轴长度：" << longAxis / 2 << std::endl;
        std::cout << "短半轴长度：" << shortAxis / 2 << std::endl;
        std::cout << "方向角：" << angle << std::endl;
    }
    return frame;
}
cv::Mat detectLinesHF(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);
    std::cout << lines.size() << std::endl;
    if (lines.size()==0)
    {
        std::cout << "检测到的是椭圆" << std::endl;
        circleDetect(image);
    }
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i line = lines[i];
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
        std::cout << "直线参数：" << std::endl;
        std::cout << "起点坐标：(" << line[0] << ", " << line[1] << ")" << std::endl;
        std::cout << "终点坐标：(" << line[2] << ", " << line[3] << ")" << std::endl;
        std::cout << "角度：" << angle << std::endl;
    }
    return image;
}
cv::Mat detectLinesH(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    std::vector<cv::Vec2f> lines;  // 注意这里的类型为 cv::Vec2f
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);  // 使用 HoughLines 函数
    std::cout << lines.size() << " 条直线被检测到" << std::endl;
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double angle = theta * 180.0 / CV_PI;
        double cosTheta = std::cos(theta);
        double sinTheta = std::sin(theta);
        double x0 = rho * cosTheta;
        double y0 = rho * sinTheta;
        cv::Point pt1(cvRound(x0 + 1000 * (-sinTheta)), cvRound(y0 + 1000 * cosTheta));
        cv::Point pt2(cvRound(x0 - 1000 * (-sinTheta)), cvRound(y0 - 1000 * cosTheta));
        cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 2);

        std::cout << "直线参数：" << std::endl;
        std::cout << "ρ：" << rho << std::endl;
        std::cout << "θ：" << theta << std::endl;
        std::cout << "角度：" << angle << std::endl;
    }

    return image;
}
cv::Mat detectLinesLSD(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    lsd->detect(grayImage, lines);
    std::cout << lines.size() << " 条直线被检测到" << std::endl;
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4f line = lines[i];
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
        std::cout << "直线参数：" << std::endl;
        std::cout << "起点坐标：(" << line[0] << ", " << line[1] << ")" << std::endl;
        std::cout << "终点坐标：(" << line[2] << ", " << line[3] << ")" << std::endl;
        std::cout << "角度：" << angle << std::endl;
    }
    return image;
}
int main()
{
    cv::VideoCapture capture("D:video.avi");
    if (!capture.isOpened()) {
        std::cout << "can not open!" << std::endl;
        return -1;
    }
    cv::Mat frame;
    while (capture.read(frame)) {
        // 定义目标分辨率
        int targetWidth = frame.cols / 4;
        int targetHeight = frame.rows / 4;
        //// 调整图像大小
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(targetWidth, targetHeight));
        /////////////
        cv::Rect roi(215, 60, 30, 120);  
        //// 在感兴趣区域内提取图像
        cv::Mat roi_frame = resized_frame(roi);
        //// 调用圆检测函数，只对感兴趣区域进行检测
        cv::Mat circle_frame = detectLinesHF(roi_frame);
        // 将检测到的椭圆框放入原始视频帧中
        circle_frame.copyTo(resized_frame(roi));
        // 展示视频
        cv::namedWindow("video", cv::WINDOW_NORMAL);
        cv::resizeWindow("video", 480, 270);
        cv::imshow("video", resized_frame);
        // 删掉的话视频会卡，原因不明
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    cv::waitKey(0);
    capture.release();
    cv::destroyAllWindows();
    return 0;
}

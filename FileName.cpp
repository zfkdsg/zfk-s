#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
cv::Mat circleDetect(cv::Mat frame) {
    cv::Mat out = frame;
    ////��ֵ�˲�
    cv::medianBlur(frame, out, 1);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
    //��Ե���
    cv::equalizeHist(out, out);
    cv::Mat edges_image;
    Canny(out, edges_image, 100, 300);
    // ��̬ѧ���� - ������
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges_image, edges_image, cv::MORPH_CLOSE, kernel);
    //��ȡ����
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    //����Բ
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

        std::cout << "��Բ������" << std::endl;
        std::cout << "Բ�����꣺(" << center.x << ", " << center.y << ")" << std::endl;
        std::cout << "�����᳤�ȣ�" << longAxis / 2 << std::endl;
        std::cout << "�̰��᳤�ȣ�" << shortAxis / 2 << std::endl;
        std::cout << "����ǣ�" << angle << std::endl;
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
        std::cout << "��⵽������Բ" << std::endl;
        circleDetect(image);
    }
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i line = lines[i];
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
        std::cout << "ֱ�߲�����" << std::endl;
        std::cout << "������꣺(" << line[0] << ", " << line[1] << ")" << std::endl;
        std::cout << "�յ����꣺(" << line[2] << ", " << line[3] << ")" << std::endl;
        std::cout << "�Ƕȣ�" << angle << std::endl;
    }
    return image;
}
cv::Mat detectLinesH(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    std::vector<cv::Vec2f> lines;  // ע�����������Ϊ cv::Vec2f
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);  // ʹ�� HoughLines ����
    std::cout << lines.size() << " ��ֱ�߱���⵽" << std::endl;
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

        std::cout << "ֱ�߲�����" << std::endl;
        std::cout << "�ѣ�" << rho << std::endl;
        std::cout << "�ȣ�" << theta << std::endl;
        std::cout << "�Ƕȣ�" << angle << std::endl;
    }

    return image;
}
cv::Mat detectLinesLSD(cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    lsd->detect(grayImage, lines);
    std::cout << lines.size() << " ��ֱ�߱���⵽" << std::endl;
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4f line = lines[i];
        cv::line(image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
        double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
        std::cout << "ֱ�߲�����" << std::endl;
        std::cout << "������꣺(" << line[0] << ", " << line[1] << ")" << std::endl;
        std::cout << "�յ����꣺(" << line[2] << ", " << line[3] << ")" << std::endl;
        std::cout << "�Ƕȣ�" << angle << std::endl;
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
        // ����Ŀ��ֱ���
        int targetWidth = frame.cols / 4;
        int targetHeight = frame.rows / 4;
        //// ����ͼ���С
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(targetWidth, targetHeight));
        /////////////
        cv::Rect roi(215, 60, 30, 120);  
        //// �ڸ���Ȥ��������ȡͼ��
        cv::Mat roi_frame = resized_frame(roi);
        //// ����Բ��⺯����ֻ�Ը���Ȥ������м��
        cv::Mat circle_frame = detectLinesHF(roi_frame);
        // ����⵽����Բ�����ԭʼ��Ƶ֡��
        circle_frame.copyTo(resized_frame(roi));
        // չʾ��Ƶ
        cv::namedWindow("video", cv::WINDOW_NORMAL);
        cv::resizeWindow("video", 480, 270);
        cv::imshow("video", resized_frame);
        // ɾ���Ļ���Ƶ�Ῠ��ԭ����
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    cv::waitKey(0);
    capture.release();
    cv::destroyAllWindows();
    return 0;
}

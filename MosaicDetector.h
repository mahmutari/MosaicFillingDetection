#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

class MosaicDetector {
private:
    cv::aruco::ArucoDetector detector_;
    int target_marker_id_;
    cv::VideoCapture camera_;
    bool is_running_;

    cv::Mat template_image_;
    cv::Mat template_lines_;
    cv::Size output_size_;

    std::map<std::pair<int, int>, cv::Scalar> grid_colors_;

    void initializeWindows();
    cv::Mat applyPerspectiveTransform(const cv::Mat& frame,
        const std::vector<cv::Point2f>& src_points);
    cv::Scalar detectCellColor(const cv::Mat& cell_bgr, const cv::Mat& cell_hsv);
    cv::Mat generateDigitalOutput(const cv::Mat& warped_frame);
    bool detectMarkers(const cv::Mat& frame,
        std::vector<std::vector<cv::Point2f>>& target_corners);
    std::vector<cv::Point2f> orderCorners(
        const std::vector<std::vector<cv::Point2f>>& markers);

public:
    MosaicDetector(const std::string& template_path, int target_marker_id, int camera_index = 0);
    ~MosaicDetector();

    void run();
    void stop();
};
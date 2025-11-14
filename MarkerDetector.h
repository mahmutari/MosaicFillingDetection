#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

class MarkerDetector {
private:
    cv::aruco::ArucoDetector detector_;
    int target_marker_id_;

    cv::Point2f getMarkerCenter(const std::vector<cv::Point2f>& corners) const;

public:
    MarkerDetector(int target_id, const cv::aruco::Dictionary& dictionary,
        const cv::aruco::DetectorParameters& params);

    bool detectMarkers(const cv::Mat& frame,
        std::vector<std::vector<cv::Point2f>>& target_corners);

    std::vector<cv::Point2f> orderCorners(
        const std::vector<std::vector<cv::Point2f>>& markers) const;
};
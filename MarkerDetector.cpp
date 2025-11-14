#include "MarkerDetector.h"
#include <utility>      // std::pair için
#include <algorithm>    // std::sort için

MarkerDetector::MarkerDetector(int target_id,
    const cv::aruco::Dictionary& dictionary,
    const cv::aruco::DetectorParameters& params)
    : target_marker_id_(target_id), detector_(dictionary, params) {
}

cv::Point2f MarkerDetector::getMarkerCenter(
    const std::vector<cv::Point2f>& corners) const {
    cv::Point2f center(0, 0);
    for (const auto& pt : corners) {
        center += pt;
    }
    return center * (1.0f / corners.size());
}

bool MarkerDetector::detectMarkers(const cv::Mat& frame,
    std::vector<std::vector<cv::Point2f>>& target_corners) {

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector_.detectMarkers(frame, corners, ids);

    target_corners.clear();
    if (!ids.empty()) {
        for (size_t i = 0; i < ids.size(); i++) {
            if (ids[i] == target_marker_id_) {
                target_corners.push_back(corners[i]);
            }
        }
    }
    return target_corners.size() == 4;
}

// Arkadaþýnýn yeni sýralama mantýðýný kullanýyoruz
std::vector<cv::Point2f> MarkerDetector::orderCorners(
    const std::vector<std::vector<cv::Point2f>>& markers) const {

    std::vector<std::pair<int, cv::Point2f>> centers;
    for (size_t i = 0; i < markers.size(); i++) {
        centers.push_back({ static_cast<int>(i), getMarkerCenter(markers[i]) });
    }

    auto temp = centers;
    std::sort(temp.begin(), temp.end(),
        [](auto& a, auto& b) { return (a.second.x + a.second.y) < (b.second.x + b.second.y); });
    int tl = temp[0].first;
    int br = temp[3].first;

    std::sort(temp.begin(), temp.end(),
        [](auto& a, auto& b) { return (a.second.x - a.second.y) > (b.second.x - b.second.y); });
    int tr = temp[0].first;
    int bl = temp[3].first;

    return {
        markers[tl][0],
        markers[tr][1],
        markers[br][2],
        markers[bl][3]
    };
}
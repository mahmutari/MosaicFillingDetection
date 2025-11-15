#include "MarkerDetector.h"
#include <algorithm>

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

std::vector<cv::Point2f> MarkerDetector::orderCorners(
    const std::vector<std::vector<cv::Point2f>>& markers) const {

    if (markers.size() != 4) {
        return {};
    }

    // Her marker'ın merkezini hesapla
    std::vector<cv::Point2f> centers;
    for (const auto& marker : markers) {
        centers.push_back(getMarkerCenter(marker));
    }

    // Merkezleri y koordinatına göre sırala (üst 2, alt 2)
    std::vector<int> indices = { 0, 1, 2, 3 };
    std::sort(indices.begin(), indices.end(),
        [&centers](int a, int b) {
            return centers[a].y < centers[b].y;
        });

    // Üst iki marker
    int top_left_idx, top_right_idx;
    if (centers[indices[0]].x < centers[indices[1]].x) {
        top_left_idx = indices[0];
        top_right_idx = indices[1];
    }
    else {
        top_left_idx = indices[1];
        top_right_idx = indices[0];
    }

    // Alt iki marker
    int bottom_left_idx, bottom_right_idx;
    if (centers[indices[2]].x < centers[indices[3]].x) {
        bottom_left_idx = indices[2];
        bottom_right_idx = indices[3];
    }
    else {
        bottom_left_idx = indices[3];
        bottom_right_idx = indices[2];
    }

    // ✅ DEĞIŞIKLIK: Her marker'dan mozaiğe en YAKIN köşeyi seç (iç köşeler)
    // Top-left marker'dan SAĞ ALT köşeyi al (mozaiğe en yakın)
    cv::Point2f tl_corner = markers[top_left_idx][0];
    for (const auto& corner : markers[top_left_idx]) {
        if (corner.x + corner.y > tl_corner.x + tl_corner.y) {  // MAKSİMUM (tersine çevirdik)
            tl_corner = corner;
        }
    }

    // Top-right marker'dan SOL ALT köşeyi al (mozaiğe en yakın)
    cv::Point2f tr_corner = markers[top_right_idx][0];
    for (const auto& corner : markers[top_right_idx]) {
        if (corner.y - corner.x > tr_corner.y - tr_corner.x) {  // MAKSİMUM (tersine çevirdik)
            tr_corner = corner;
        }
    }

    // Bottom-right marker'dan SOL ÜST köşeyi al (mozaiğe en yakın)
    cv::Point2f br_corner = markers[bottom_right_idx][0];
    for (const auto& corner : markers[bottom_right_idx]) {
        if (corner.x + corner.y < br_corner.x + br_corner.y) {  // MİNİMUM (tersine çevirdik)
            br_corner = corner;
        }
    }

    // Bottom-left marker'dan SAĞ ÜST köşeyi al (mozaiğe en yakın)
    cv::Point2f bl_corner = markers[bottom_left_idx][0];
    for (const auto& corner : markers[bottom_left_idx]) {
        if (corner.x - corner.y > bl_corner.x - bl_corner.y) {  // MAKSİMUM (tersine çevirdik)
            bl_corner = corner;
        }
    }

    return { tl_corner, tr_corner, br_corner, bl_corner };
}
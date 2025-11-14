#include "MosaicDetector.h"

MosaicDetector::MosaicDetector(const std::string& template_path,
    int target_marker_id,
    int camera_index)
    : target_marker_id_(target_marker_id), is_running_(false) {

    // Template yükle
    template_image_ = cv::imread(template_path);
    if (template_image_.empty()) {
        throw std::runtime_error("Failed to load template: " + template_path);
    }
    output_size_ = template_image_.size();

    // Template çizgilerini çýkar
    cv::Mat gray;
    cv::cvtColor(template_image_, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, template_lines_, 200, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(template_lines_, template_lines_, kernel);

    // ArUco detector
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    detector_ = cv::aruco::ArucoDetector(dictionary, params);

    // Kamera
    camera_.open(camera_index);
    if (!camera_.isOpened()) {
        throw std::runtime_error("Failed to open camera!");
    }
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    initializeWindows();
}

MosaicDetector::~MosaicDetector() {
    stop();
}

void MosaicDetector::initializeWindows() {
    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Warped", cv::WINDOW_NORMAL);
    cv::namedWindow("Digital Mosaic", cv::WINDOW_NORMAL);
}

bool MosaicDetector::detectMarkers(const cv::Mat& frame,
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

std::vector<cv::Point2f> MosaicDetector::orderCorners(
    const std::vector<std::vector<cv::Point2f>>& markers) {

    // Marker merkezlerini hesapla
    std::vector<std::pair<int, cv::Point2f>> centers;
    for (size_t i = 0; i < markers.size(); i++) {
        cv::Point2f center(0, 0);
        for (const auto& pt : markers[i]) {
            center += pt;
        }
        center *= (1.0f / markers[i].size());
        centers.push_back({ i, center });
    }

    // Top-left, top-right, bottom-right, bottom-left
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

cv::Mat MosaicDetector::applyPerspectiveTransform(
    const cv::Mat& frame,
    const std::vector<cv::Point2f>& src_points) {

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(output_size_.width - 1, 0),
        cv::Point2f(output_size_.width - 1, output_size_.height - 1),
        cv::Point2f(0, output_size_.height - 1)
    };

    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(frame, warped, M, output_size_,
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    return warped;
}

cv::Scalar MosaicDetector::detectCellColor(const cv::Mat& cell_bgr, const cv::Mat& cell_hsv) {
    // Renk sayýmý
    int red = 0, orange = 0, yellow = 0, green = 0, blue = 0, purple = 0, white = 0;

    for (int y = 0; y < cell_hsv.rows; y++) {
        for (int x = 0; x < cell_hsv.cols; x++) {
            cv::Vec3b hsv = cell_hsv.at<cv::Vec3b>(y, x);
            cv::Vec3b bgr = cell_bgr.at<cv::Vec3b>(y, x);

            int h = hsv[0], s = hsv[1], v = hsv[2];
            int b = bgr[0], g = bgr[1], r = bgr[2];

            // Beyaz/gri/siyah
            if (s < 40 || v < 60 || v > 240) {
                white++;
                continue;
            }

            // Renkler
            if (((h <= 10) || (h >= 170)) && r > 140 && r > g * 1.5 && r > b * 1.5) {
                red++;
            }
            else if (h >= 11 && h <= 25 && r > 150 && g > 60 && r > g * 1.2) {
                orange++;
            }
            else if (h >= 26 && h <= 40 && r > 150 && g > 150) {
                yellow++;
            }
            else if (h >= 45 && h <= 85 && g > 130 && g > r * 1.3) {
                green++;
            }
            else if (h >= 100 && h <= 130 && b > 130 && b > r * 1.2) {
                blue++;
            }
            else if (h >= 135 && h <= 160 && r > 110 && b > 110) {
                purple++;
            }
        }
    }

    // En fazla olan rengi döndür
    int max_count = std::max({ red, orange, yellow, green, blue, purple });

    if (max_count < 50) return cv::Scalar(255, 255, 255);  // Beyaz

    if (red == max_count) return cv::Scalar(0, 0, 255);
    if (orange == max_count) return cv::Scalar(0, 165, 255);
    if (yellow == max_count) return cv::Scalar(0, 255, 255);
    if (green == max_count) return cv::Scalar(0, 255, 0);
    if (blue == max_count) return cv::Scalar(255, 0, 0);
    if (purple == max_count) return cv::Scalar(255, 0, 255);

    return cv::Scalar(255, 255, 255);
}

cv::Mat MosaicDetector::generateDigitalOutput(const cv::Mat& warped_frame) {
    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    // Boþ digital output
    cv::Mat digital = cv::Mat(output_size_, CV_8UC3, cv::Scalar(255, 255, 255));

    // Þablon contourlarýný kullan
    cv::Mat gray_template;
    cv::cvtColor(template_image_, gray_template, cv::COLOR_BGR2GRAY);
    cv::Mat template_edges;
    cv::threshold(gray_template, template_edges, 200, 255, cv::THRESH_BINARY_INV);

    // Morfolojik iþlem
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(template_edges, template_edges, kernel);

    // Ters çevir
    cv::Mat inverted;
    cv::bitwise_not(template_edges, inverted);

    // Contourlarý bul
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(inverted.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "Found " << contours.size() << " contours from template" << std::endl;

    // Her contour için
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 200 || area > output_size_.area() * 0.2) continue;

        // Contour maskesi
        cv::Mat mask = cv::Mat::zeros(output_size_, CV_8UC1);
        cv::drawContours(mask, contours, i, cv::Scalar(255), cv::FILLED);

        cv::Rect bbox = cv::boundingRect(contours[i]);

        // Bounds check
        if (bbox.x < 0 || bbox.y < 0 ||
            bbox.x + bbox.width > warped_frame.cols ||
            bbox.y + bbox.height > warped_frame.rows) continue;

        // Renk sayýmý
        int red = 0, orange = 0, yellow = 0, green = 0, blue = 0, purple = 0;
        int total = 0;

        // ROI üzerinde çalýþ
        for (int y = bbox.y; y < bbox.y + bbox.height; y++) {
            for (int x = bbox.x; x < bbox.x + bbox.width; x++) {
                if (mask.at<uchar>(y, x) == 0) continue;

                cv::Vec3b hsv = hsv_warped.at<cv::Vec3b>(y, x);
                cv::Vec3b bgr = warped_frame.at<cv::Vec3b>(y, x);

                int h = hsv[0], s = hsv[1], v = hsv[2];
                int b = bgr[0], g = bgr[1], r = bgr[2];

                // Beyaz/gri/siyah - atla
                if (s < 40 || v < 60 || v > 240) continue;

                total++;

                // Renkler
                if (((h <= 10) || (h >= 170)) && r > 140 && r > g * 1.5 && r > b * 1.5) {
                    red++;
                }
                else if (h >= 11 && h <= 25 && r > 150 && g > 60 && r > g * 1.2) {
                    orange++;
                }
                else if (h >= 26 && h <= 40 && r > 150 && g > 150) {
                    yellow++;
                }
                else if (h >= 45 && h <= 85 && g > 130 && g > r * 1.3) {
                    green++;
                }
                else if (h >= 100 && h <= 130 && b > 130 && b > r * 1.2) {
                    blue++;
                }
                else if (h >= 135 && h <= 160 && r > 110 && b > 110) {
                    purple++;
                }
            }
        }

        // Minimum threshold
        if (total < 50) continue;

        // En fazla olan renk
        int max_count = std::max({ red, orange, yellow, green, blue, purple });

        cv::Scalar color = cv::Scalar(255, 255, 255);

        if (red == max_count && red > 0) color = cv::Scalar(0, 0, 255);
        else if (orange == max_count && orange > 0) color = cv::Scalar(0, 165, 255);
        else if (yellow == max_count && yellow > 0) color = cv::Scalar(0, 255, 255);
        else if (green == max_count && green > 0) color = cv::Scalar(0, 255, 0);
        else if (blue == max_count && blue > 0) color = cv::Scalar(255, 0, 0);
        else if (purple == max_count && purple > 0) color = cv::Scalar(255, 0, 255);

        // Smooth transition
        auto key = std::make_pair((int)i, 0);
        if (grid_colors_.find(key) != grid_colors_.end()) {
            cv::Scalar prev = grid_colors_[key];
            if (prev != color) {
                // Renk deðiþimi varsa smooth geçiþ
                color = color * 0.7 + prev * 0.3;
            }
        }
        grid_colors_[key] = color;

        // Doldur
        digital.setTo(color, mask);
    }

    // Çizgileri ekle
    digital.setTo(cv::Scalar(0, 0, 0), template_edges);

    return digital;
}

void MosaicDetector::run() {
    is_running_ = true;
    std::cout << "Mosaic Detector started. Press 'q' to quit." << std::endl;

    while (is_running_) {
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) break;

        std::vector<std::vector<cv::Point2f>> target_corners;
        bool found = detectMarkers(frame, target_corners);

        cv::Mat display = frame.clone();

        if (found) {
            auto corners = orderCorners(target_corners);

            // Draw corners
            for (size_t i = 0; i < corners.size(); i++) {
                cv::circle(display, corners[i], 8, cv::Scalar(0, 255, 0), -1);
            }

            cv::Mat warped = applyPerspectiveTransform(frame, corners);
            cv::Mat digital = generateDigitalOutput(warped);

            cv::imshow("Warped", warped);
            cv::imshow("Digital Mosaic", digital);
        }

        cv::imshow("Live Video", display);

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    stop();
}

void MosaicDetector::stop() {
    is_running_ = false;
    if (camera_.isOpened()) camera_.release();
    cv::destroyAllWindows();
}
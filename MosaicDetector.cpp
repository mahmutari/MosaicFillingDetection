#include "MosaicDetector.h"
#include <stdexcept>
#include <iostream>

// ==================== MOSAIC DETECTOR (ORKESTRA ÞEFÝ) ====================

MosaicDetector::MosaicDetector(const std::string& template_path,
    int target_marker_id,
    int camera_index)
    : is_running_(false) {

    // 1. Modülleri baþlat
    template_processor_ = std::make_unique<TemplateProcessor>(template_path);
    color_detector_ = std::make_unique<ColorDetector>();

    // 2. ArUco algýlayýcýyý (arkadaþýnýn kodu gibi) baþlat
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    marker_detector_ = std::make_unique<MarkerDetector>(target_marker_id, dictionary, params);

    // 3. Renk geçmiþini kontur sayýsýna göre ayarla
    color_histories_.resize(template_processor_->getContours().size());

    // 4. Kamerayý aç
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

// Bu fonksiyon PerspectiveWarper sýnýfýna da taþýnabilir, 
// ama þimdilik burada kalmasý daha basit.
cv::Mat MosaicDetector::applyPerspectiveTransform(
    const cv::Mat& frame,
    const std::vector<cv::Point2f>& src_points) {

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(template_processor_->getOutputSize().width - 1, 0),
        cv::Point2f(template_processor_->getOutputSize().width - 1, template_processor_->getOutputSize().height - 1),
        cv::Point2f(0, template_processor_->getOutputSize().height - 1)
    };

    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(frame, warped, M, template_processor_->getOutputSize(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // Görüntü netliði için BLUR'u kaldýrdýk
    return warped;
}

// Bizim "DOLU" ve "UNUTAN" mantýðýmýzý kullanan GÜNCEL fonksiyon
cv::Mat MosaicDetector::generateDigitalOutput(const cv::Mat& warped_frame) {
    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    cv::Mat digital_output(template_processor_->getOutputSize(), CV_8UC3, cv::Scalar(255, 255, 255));
    const auto& contours = template_processor_->getContours();

    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Mat mask = cv::Mat::zeros(warped_frame.size(), CV_8U);
        cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

        // Rengi bul
        cv::Scalar dominant_color = color_detector_->detectDominantColor(warped_frame, hsv_warped, mask);

        cv::Scalar color_to_draw;
        bool is_white = (dominant_color[0] == 255 && dominant_color[1] == 255 && dominant_color[2] == 255);

        // "Unutma" mantýðý
        if (is_white) {
            color_histories_[i].clear();
            color_to_draw = cv::Scalar(255, 255, 255);
        }
        else {
            color_histories_[i].addColor(dominant_color);
            color_to_draw = color_histories_[i].getStableColor();
        }

        // Doldur
        cv::drawContours(digital_output, contours, static_cast<int>(i), color_to_draw, cv::FILLED);
    }

    // Çizgileri ekle
    digital_output.setTo(cv::Scalar(0, 0, 0), template_processor_->getTemplateLines());
    return digital_output;
}

void MosaicDetector::run() {
    is_running_ = true;
    std::cout << "Mosaic Detector started. Press 'q' to quit." << std::endl;

    while (is_running_) {
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) break;

        processFrame(frame); // Her kareyi iþle

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    stop();
}

void MosaicDetector::processFrame(cv::Mat& frame) {
    std::vector<std::vector<cv::Point2f>> target_corners;
    bool found = marker_detector_->detectMarkers(frame, target_corners);

    cv::Mat display = frame.clone();

    if (found) {
        auto corners = marker_detector_->orderCorners(target_corners);

        // Köþeleri çiz
        for (size_t i = 0; i < corners.size(); i++) {
            cv::circle(display, corners[i], 8, cv::Scalar(0, 255, 0), -1);
        }

        cv::Mat warped = applyPerspectiveTransform(frame, corners);
        cv::Mat digital = generateDigitalOutput(warped);

        cv::imshow("Warped", warped);
        cv::imshow("Digital Mosaic", digital);
    }

    cv::imshow("Live Video", display);
}

void MosaicDetector::stop() {
    is_running_ = false;
    if (camera_.isOpened()) camera_.release();
    cv::destroyAllWindows();
}
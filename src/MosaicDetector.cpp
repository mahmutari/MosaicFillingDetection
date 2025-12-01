#include "MosaicDetector.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>

// Minimum fill ratio - bunun altındaki değerler beyaz olarak kabul edilir
const float MIN_FILL_RATIO_THRESHOLD = 0.15f;  // %15

MosaicDetector::MosaicDetector(const std::string& template_path,
    int target_marker_id,
    int camera_index)
    : is_running_(false) {

    template_processor_ = std::make_unique<TemplateProcessor>(template_path);
    color_detector_ = std::make_unique<ColorDetector>();

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    marker_detector_ = std::make_unique<MarkerDetector>(target_marker_id, dictionary, params);

    size_t num_contours = template_processor_->getContours().size();
    color_histories_.resize(num_contours);
    ratio_histories_.resize(num_contours, 0.0f);

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

cv::Point MosaicDetector::calculateContourCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        cv::Rect br = cv::boundingRect(contour);
        return cv::Point(br.x + br.width / 2, br.y + br.height / 2);
    }
    return cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
}

cv::Mat MosaicDetector::applyPerspectiveTransform(
    const cv::Mat& frame,
    const std::vector<cv::Point2f>& src_points) {

    float width1 = cv::norm(src_points[1] - src_points[0]);
    float width2 = cv::norm(src_points[2] - src_points[3]);
    float height1 = cv::norm(src_points[3] - src_points[0]);
    float height2 = cv::norm(src_points[2] - src_points[1]);

    int warp_width = static_cast<int>((width1 + width2) / 2.0f);
    int warp_height = static_cast<int>((height1 + height2) / 2.0f);

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(warp_width - 1, 0),
        cv::Point2f(warp_width - 1, warp_height - 1),
        cv::Point2f(0, warp_height - 1)
    };

    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(frame, warped, M, cv::Size(warp_width, warp_height),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    
    return warped;
}

cv::Mat MosaicDetector::generateDigitalOutput(const cv::Mat& warped_frame,
    std::vector<PatchInfo>& patch_infos) {
    patch_infos.clear();

    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    cv::Size template_size = template_processor_->getOutputSize();
    cv::Size warped_size = warped_frame.size();

    float scale_x = static_cast<float>(warped_size.width) / template_size.width;
    float scale_y = static_cast<float>(warped_size.height) / template_size.height;

    cv::Mat digital_output(warped_size, CV_8UC3, cv::Scalar(255, 255, 255));
    const auto& original_contours = template_processor_->getContours();

    for (size_t i = 0; i < original_contours.size(); ++i) {
        std::vector<cv::Point> scaled_contour;
        for (const auto& pt : original_contours[i]) {
            scaled_contour.push_back(cv::Point(
                static_cast<int>(pt.x * scale_x),
                static_cast<int>(pt.y * scale_y)
            ));
        }

        cv::Mat mask_full = cv::Mat::zeros(warped_size, CV_8U);
        std::vector<std::vector<cv::Point>> contour_vec = { scaled_contour };
        cv::drawContours(mask_full, contour_vec, 0, cv::Scalar(255), cv::FILLED);

        // Orijinal erode - sadece siyah çizgileri hariç tutmak için
        cv::Mat mask_eroded;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mask_full, mask_eroded, kernel, cv::Point(-1, -1), 2);

        // Ratio bilgisi ile renk tespiti
        ColorDetectionResult detection = color_detector_->detectColorWithRatio(
            warped_frame, hsv_warped, mask_eroded);

        cv::Scalar color_to_draw;
        float current_ratio = detection.fill_ratio;
        std::string current_color_name = detection.color_name;

        // Beyaz kontrolü VEYA çok düşük fill ratio
        bool is_white = (detection.color[0] == 255 &&
            detection.color[1] == 255 &&
            detection.color[2] == 255);

        bool is_below_threshold = (current_ratio < MIN_FILL_RATIO_THRESHOLD);

        if (is_white || is_below_threshold) {
            // Beyaz olarak kabul et
            color_histories_[i].clear();
            color_to_draw = cv::Scalar(255, 255, 255);
            ratio_histories_[i] = 0.0f;
            current_color_name = "White";
            current_ratio = 0.0f;
        }
        else {
            color_histories_[i].addColor(detection.color);
            color_to_draw = color_histories_[i].getStableColor();
            // Ratio smoothing
            ratio_histories_[i] = ratio_histories_[i] * 0.7f + current_ratio * 0.3f;
        }

        cv::drawContours(digital_output, contour_vec, 0, color_to_draw, cv::FILLED);

        // Patch bilgisini kaydet
        PatchInfo info;
        info.patch_id = static_cast<int>(i);
        info.color_name = current_color_name;
        info.fill_ratio = ratio_histories_[i];
        info.centroid = calculateContourCentroid(scaled_contour);
        patch_infos.push_back(info);
    }

    cv::Mat template_lines = template_processor_->getTemplateLines();
    cv::Mat scaled_lines;
    cv::resize(template_lines, scaled_lines, warped_size, 0, 0, cv::INTER_NEAREST);
    digital_output.setTo(cv::Scalar(0, 0, 0), scaled_lines);

    return digital_output;
}

void MosaicDetector::drawRatioInfo(cv::Mat& image, const std::vector<PatchInfo>& patch_infos) {
    for (const auto& info : patch_infos) {
        // Sadece renkli ve eşik üstü patch'ler için ratio göster
        if (info.color_name == "White" || info.fill_ratio < MIN_FILL_RATIO_THRESHOLD) continue;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(0) << (info.fill_ratio * 100) << "%";
        std::string ratio_text = ss.str();

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.35;
        int thickness = 1;
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(ratio_text, font, font_scale, thickness, &baseline);

        cv::Point text_pos(
            info.centroid.x - text_size.width / 2,
            info.centroid.y + text_size.height / 2
        );

        // Arka plan
        cv::Rect bg_rect(
            text_pos.x - 2,
            text_pos.y - text_size.height - 2,
            text_size.width + 4,
            text_size.height + 4
        );

        if (bg_rect.x >= 0 && bg_rect.y >= 0 &&
            bg_rect.x + bg_rect.width < image.cols &&
            bg_rect.y + bg_rect.height < image.rows) {

            cv::Mat roi = image(bg_rect);
            cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
            cv::addWeighted(roi, 0.5, overlay, 0.5, 0, roi);

            cv::putText(image, ratio_text, text_pos, font, font_scale,
                cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }
}

void MosaicDetector::run() {
    is_running_ = true;
    std::cout << "Mosaic Detector started. Press 'q' to quit." << std::endl;

    while (is_running_) {
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) break;

        processFrame(frame);

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

        for (size_t i = 0; i < corners.size(); i++) {
            cv::circle(display, corners[i], 8, cv::Scalar(0, 255, 0), -1);
        }

        cv::Mat warped = applyPerspectiveTransform(frame, corners);

        std::vector<PatchInfo> patch_infos;
        cv::Mat digital = generateDigitalOutput(warped, patch_infos);

        // Ratio bilgisini çiz
        drawRatioInfo(digital, patch_infos);

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
#include "MosaicDetector.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

// Minimum fill ratio - bunun altındaki değerler beyaz olarak kabul edilir
const float MIN_FILL_RATIO_THRESHOLD = 0.15f;  // %15

// Template değişimi için gereken tutarlı frame sayısı
const int TEMPLATE_SWITCH_THRESHOLD = 10;

MosaicDetector::MosaicDetector(const std::vector<std::string>& template_paths,
    const std::vector<std::string>& template_names,
    int target_marker_id,
    int camera_index)
    : is_running_(false), current_rotation_(0), current_template_index_(0),
    detected_template_index_(0), template_vote_count_(0) {

    if (template_paths.empty()) {
        throw std::runtime_error("At least one template path is required!");
    }

    template_paths_ = template_paths;
    template_names_ = template_names;

    // İsim sayısı yeterli değilse varsayılan isimler ekle
    while (template_names_.size() < template_paths_.size()) {
        template_names_.push_back("Template " + std::to_string(template_names_.size() + 1));
    }

    // Tüm template'leri yükle
    for (size_t i = 0; i < template_paths.size(); ++i) {
        try {
            auto processor = std::make_unique<TemplateProcessor>(template_paths[i]);
            template_processors_.push_back(std::move(processor));

            // Her template için history oluştur
            size_t num_contours = template_processors_.back()->getContours().size();
            all_color_histories_.push_back(std::vector<ColorHistory>(num_contours));
            all_ratio_histories_.push_back(std::vector<float>(num_contours, 0.0f));

            std::cout << "Template loaded: " << template_names_[i]
                << " (" << template_paths[i] << ")" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Could not load template " << template_paths[i]
                << ": " << e.what() << std::endl;
        }
    }

    if (template_processors_.empty()) {
        throw std::runtime_error("No valid templates could be loaded!");
    }

    color_detector_ = std::make_unique<ColorDetector>();

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    marker_detector_ = std::make_unique<MarkerDetector>(target_marker_id, dictionary, params);

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

void MosaicDetector::switchTemplate(int index) {
    if (index >= 0 && index < static_cast<int>(template_processors_.size()) &&
        index != current_template_index_) {
        current_template_index_ = index;
        std::cout << "Auto-switched to: " << template_names_[index] << std::endl;
    }
}

void MosaicDetector::resetHistories(int index) {
    if (index >= 0 && index < static_cast<int>(all_color_histories_.size())) {
        for (auto& history : all_color_histories_[index]) {
            history.clear();
        }
        std::fill(all_ratio_histories_[index].begin(),
            all_ratio_histories_[index].end(), 0.0f);
    }
}

cv::Point MosaicDetector::calculateContourCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        cv::Rect br = cv::boundingRect(contour);
        return cv::Point(br.x + br.width / 2, br.y + br.height / 2);
    }
    return cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
}

int MosaicDetector::detectRotation(const std::vector<std::vector<cv::Point2f>>& markers) {
    if (markers.size() != 4) return 0;

    const auto& first_marker = markers[0];
    cv::Point2f marker_center(0, 0);
    for (const auto& pt : first_marker) {
        marker_center += pt;
    }
    marker_center *= 0.25f;

    cv::Point2f corner0_dir = first_marker[0] - marker_center;

    float angle = std::atan2(corner0_dir.y, corner0_dir.x) * 180.0f / CV_PI;

    if (angle < 0) angle += 360.0f;

    if (angle >= 180.0f && angle < 270.0f) {
        return 0;
    }
    else if (angle >= 270.0f && angle < 360.0f) {
        return 90;
    }
    else if (angle >= 0.0f && angle < 90.0f) {
        return 180;
    }
    else {
        return 270;
    }
}

cv::Mat MosaicDetector::rotateImage(const cv::Mat& image, int rotation) {
    if (rotation == 0) {
        return image.clone();
    }

    cv::Mat rotated;

    if (rotation == 90) {
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    }
    else if (rotation == 180) {
        cv::rotate(image, rotated, cv::ROTATE_180);
    }
    else if (rotation == 270) {
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    else {
        return image.clone();
    }

    return rotated;
}

cv::Mat MosaicDetector::rotateImageInverse(const cv::Mat& image, int rotation) {
    if (rotation == 0) {
        return image.clone();
    }

    cv::Mat rotated;

    if (rotation == 90) {
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    else if (rotation == 180) {
        cv::rotate(image, rotated, cv::ROTATE_180);
    }
    else if (rotation == 270) {
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    }
    else {
        return image.clone();
    }

    return rotated;
}

double MosaicDetector::calculateTemplateSimilarity(const cv::Mat& warped_lines, int template_index) {
    if (template_index < 0 || template_index >= static_cast<int>(template_processors_.size())) {
        return 0.0;
    }

    // Template'in siyah çizgilerini al
    cv::Mat template_lines = template_processors_[template_index]->getTemplateLines();

    // Aynı boyuta getir
    cv::Mat template_resized;
    cv::resize(template_lines, template_resized, warped_lines.size(), 0, 0, cv::INTER_NEAREST);

    // Her iki maske de binary olmalı
    cv::Mat warped_binary, template_binary;

    if (warped_lines.channels() > 1) {
        cv::cvtColor(warped_lines, warped_binary, cv::COLOR_BGR2GRAY);
    }
    else {
        warped_binary = warped_lines.clone();
    }

    if (template_resized.channels() > 1) {
        cv::cvtColor(template_resized, template_binary, cv::COLOR_BGR2GRAY);
    }
    else {
        template_binary = template_resized.clone();
    }

    // Binary threshold
    cv::threshold(warped_binary, warped_binary, 127, 255, cv::THRESH_BINARY);
    cv::threshold(template_binary, template_binary, 127, 255, cv::THRESH_BINARY);

    // Intersection over Union (IoU) benzeri metrik
    cv::Mat intersection, union_mask;
    cv::bitwise_and(warped_binary, template_binary, intersection);
    cv::bitwise_or(warped_binary, template_binary, union_mask);

    double intersection_count = cv::countNonZero(intersection);
    double union_count = cv::countNonZero(union_mask);

    if (union_count == 0) return 0.0;

    return intersection_count / union_count;
}

int MosaicDetector::detectTemplate(const cv::Mat& warped_normalized) {
    if (template_processors_.size() <= 1) {
        return 0;  // Tek template varsa o
    }

    // Warped görüntüden siyah çizgileri çıkar
    cv::Mat gray;
    cv::cvtColor(warped_normalized, gray, cv::COLOR_BGR2GRAY);

    // Siyah çizgileri bul (düşük değerli pikseller)
    cv::Mat warped_lines;
    cv::threshold(gray, warped_lines, 60, 255, cv::THRESH_BINARY_INV);

    // Gürültüyü azalt
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(warped_lines, warped_lines, cv::MORPH_CLOSE, kernel);

    // Her template ile karşılaştır
    double best_similarity = 0.0;
    int best_index = 0;

    for (size_t i = 0; i < template_processors_.size(); ++i) {
        double similarity = calculateTemplateSimilarity(warped_lines, static_cast<int>(i));

        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = static_cast<int>(i);
        }
    }

    return best_index;
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

    auto& template_processor = template_processors_[current_template_index_];
    auto& color_histories = all_color_histories_[current_template_index_];
    auto& ratio_histories = all_ratio_histories_[current_template_index_];

    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    cv::Size template_size = template_processor->getOutputSize();
    cv::Size warped_size = warped_frame.size();

    float scale_x = static_cast<float>(warped_size.width) / template_size.width;
    float scale_y = static_cast<float>(warped_size.height) / template_size.height;

    cv::Mat digital_output(warped_size, CV_8UC3, cv::Scalar(255, 255, 255));
    const auto& original_contours = template_processor->getContours();

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

        cv::Mat mask_eroded;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mask_full, mask_eroded, kernel, cv::Point(-1, -1), 2);

        ColorDetectionResult detection = color_detector_->detectColorWithRatio(
            warped_frame, hsv_warped, mask_eroded);

        cv::Scalar color_to_draw;
        float current_ratio = detection.fill_ratio;
        std::string current_color_name = detection.color_name;

        bool is_white = (detection.color[0] == 255 &&
            detection.color[1] == 255 &&
            detection.color[2] == 255);

        bool is_below_threshold = (current_ratio < MIN_FILL_RATIO_THRESHOLD);

        if (is_white || is_below_threshold) {
            color_histories[i].clear();
            color_to_draw = cv::Scalar(255, 255, 255);
            ratio_histories[i] = 0.0f;
            current_color_name = "White";
            current_ratio = 0.0f;
        }
        else {
            color_histories[i].addColor(detection.color);
            color_to_draw = color_histories[i].getStableColor();
            ratio_histories[i] = ratio_histories[i] * 0.7f + current_ratio * 0.3f;
        }

        cv::drawContours(digital_output, contour_vec, 0, color_to_draw, cv::FILLED);

        PatchInfo info;
        info.patch_id = static_cast<int>(i);
        info.color_name = current_color_name;
        info.fill_ratio = ratio_histories[i];
        info.centroid = calculateContourCentroid(scaled_contour);
        patch_infos.push_back(info);
    }

    cv::Mat template_lines = template_processor->getTemplateLines();
    cv::Mat scaled_lines;
    cv::resize(template_lines, scaled_lines, warped_size, 0, 0, cv::INTER_NEAREST);
    digital_output.setTo(cv::Scalar(0, 0, 0), scaled_lines);

    return digital_output;
}

void MosaicDetector::drawRatioInfo(cv::Mat& image, const std::vector<PatchInfo>& patch_infos) {
    for (const auto& info : patch_infos) {
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
    std::cout << "\n=== Mosaic Detector ===" << std::endl;
    std::cout << "Templates loaded: " << template_processors_.size() << std::endl;
    for (size_t i = 0; i < template_names_.size() && i < template_processors_.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << template_names_[i] << std::endl;
    }
    std::cout << "\nAutomatic template detection: ENABLED" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  'r' - Reset current template histories" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << "\nWaiting for mosaic..." << std::endl;

    while (is_running_) {
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) break;

        processFrame(frame);

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
        else if (key == 'r') {
            resetHistories(current_template_index_);
            std::cout << "Histories reset for " << template_names_[current_template_index_] << std::endl;
        }
    }
    stop();
}

void MosaicDetector::processFrame(cv::Mat& frame) {
    std::vector<std::vector<cv::Point2f>> target_corners;
    bool found = marker_detector_->detectMarkers(frame, target_corners);

    cv::Mat display = frame.clone();

    if (found) {
        int detected_rotation = detectRotation(target_corners);

        if (detected_rotation != current_rotation_) {
            rotation_vote_count_++;
            if (rotation_vote_count_ > 5) {
                current_rotation_ = detected_rotation;
                rotation_vote_count_ = 0;
            }
        }
        else {
            rotation_vote_count_ = 0;
        }

        auto corners = marker_detector_->orderCorners(target_corners);

        for (size_t i = 0; i < corners.size(); i++) {
            cv::circle(display, corners[i], 8, cv::Scalar(0, 255, 0), -1);
        }

        cv::Mat warped = applyPerspectiveTransform(frame, corners);
        cv::Mat warped_normalized = rotateImageInverse(warped, current_rotation_);

        // Otomatik template algılama
        int detected_template = detectTemplate(warped_normalized);

        if (detected_template != detected_template_index_) {
            detected_template_index_ = detected_template;
            template_vote_count_ = 1;
        }
        else {
            template_vote_count_++;
        }

        // Belirli sayıda tutarlı algılama sonrası template değiştir
        if (template_vote_count_ >= TEMPLATE_SWITCH_THRESHOLD &&
            detected_template_index_ != current_template_index_) {
            switchTemplate(detected_template_index_);
            template_vote_count_ = 0;
        }

        std::vector<PatchInfo> patch_infos;
        cv::Mat digital_normalized = generateDigitalOutput(warped_normalized, patch_infos);

        // Önce döndür, sonra ratio bilgisini çiz (yazılar düz kalır)
        cv::Mat digital_rotated = rotateImage(digital_normalized, current_rotation_);

        // Patch centroid'lerini de döndür
        std::vector<PatchInfo> rotated_patch_infos = patch_infos;
        cv::Size norm_size = digital_normalized.size();
        cv::Size rot_size = digital_rotated.size();

        for (auto& info : rotated_patch_infos) {
            cv::Point old_pt = info.centroid;
            cv::Point new_pt;

            if (current_rotation_ == 90) {
                new_pt.x = norm_size.height - 1 - old_pt.y;
                new_pt.y = old_pt.x;
            }
            else if (current_rotation_ == 180) {
                new_pt.x = norm_size.width - 1 - old_pt.x;
                new_pt.y = norm_size.height - 1 - old_pt.y;
            }
            else if (current_rotation_ == 270) {
                new_pt.x = old_pt.y;
                new_pt.y = norm_size.width - 1 - old_pt.x;
            }
            else {
                new_pt = old_pt;
            }

            info.centroid = new_pt;
        }

        drawRatioInfo(digital_rotated, rotated_patch_infos);

        // Template bilgisini göster
        std::string template_info = template_names_[current_template_index_];
        cv::putText(digital_rotated, template_info, cv::Point(10, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(digital_rotated, template_info, cv::Point(10, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        cv::imshow("Warped", warped);
        cv::imshow("Digital Mosaic", digital_rotated);
    }

    cv::imshow("Live Video", display);
}

void MosaicDetector::stop() {
    is_running_ = false;
    if (camera_.isOpened()) camera_.release();
    cv::destroyAllWindows();
}
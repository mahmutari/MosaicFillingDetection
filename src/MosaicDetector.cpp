#include "MosaicDetector.h"
#include <stdexcept>
#include <iostream>

// ==================== MOSAIC DETECTOR (ORKESTRA ŞEFİ) ====================

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

    color_histories_.resize(template_processor_->getContours().size());

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

cv::Mat MosaicDetector::applyPerspectiveTransform(
    const cv::Mat& frame,
    const std::vector<cv::Point2f>& src_points) {

    // ArUco marker'ların oluşturduğu dikdörtgenin boyutunu hesapla
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

    // ✅ Kenarlardan %5 kırp (marker çerçevelerini at)
    int crop_x = static_cast<int>(warp_width * 0.08f);
    int crop_y = static_cast<int>(warp_height * 0.08f);

    cv::Rect crop_rect(
        crop_x,
        crop_y,
        warp_width - 2 * crop_x,
        warp_height - 2 * crop_y
    );

    // Kırpma yapılabilir mi kontrol et
    if (crop_rect.width > 0 && crop_rect.height > 0 &&
        crop_rect.x >= 0 && crop_rect.y >= 0 &&
        crop_rect.x + crop_rect.width <= warped.cols &&
        crop_rect.y + crop_rect.height <= warped.rows) {
        warped = warped(crop_rect);
    }

    return warped;
}

cv::Mat MosaicDetector::generateDigitalOutput(const cv::Mat& warped_frame) {
    // Renk tespitini warped'in KENDİ boyutunda yap
    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    // Contour'ları warped'in boyutuna scale et
    cv::Size template_size = template_processor_->getOutputSize();
    cv::Size warped_size = warped_frame.size();

    float scale_x = static_cast<float>(warped_size.width) / template_size.width;
    float scale_y = static_cast<float>(warped_size.height) / template_size.height;

    cv::Mat digital_output(warped_size, CV_8UC3, cv::Scalar(255, 255, 255));
    const auto& original_contours = template_processor_->getContours();

    for (size_t i = 0; i < original_contours.size(); ++i) {
        // Contour'u warped boyutuna scale et
        std::vector<cv::Point> scaled_contour;
        for (const auto& pt : original_contours[i]) {
            scaled_contour.push_back(cv::Point(
                static_cast<int>(pt.x * scale_x),
                static_cast<int>(pt.y * scale_y)
            ));
        }

        // ✅ YENİ: Renk tespiti için KÜÇÜLTÜLMÜŞ mask oluştur
        cv::Mat mask_full = cv::Mat::zeros(warped_size, CV_8U);
        std::vector<std::vector<cv::Point>> contour_vec = { scaled_contour };
        cv::drawContours(mask_full, contour_vec, 0, cv::Scalar(255), cv::FILLED);

        // Mask'i küçült (erode) - sadece patch'in ortasını al
        cv::Mat mask_eroded;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mask_full, mask_eroded, kernel, cv::Point(-1, -1), 2);  // 2 kez erode

        // Renk tespiti küçültülmüş mask ile
        cv::Scalar dominant_color = color_detector_->detectDominantColor(warped_frame, hsv_warped, mask_eroded);

        cv::Scalar color_to_draw;
        bool is_white = (dominant_color[0] == 255 && dominant_color[1] == 255 && dominant_color[2] == 255);

        if (is_white) {
            color_histories_[i].clear();
            color_to_draw = cv::Scalar(255, 255, 255);
        }
        else {
            color_histories_[i].addColor(dominant_color);
            color_to_draw = color_histories_[i].getStableColor();
        }

        // ✅ Doldururken ORIJINAL (tam) contour'u kullan
        cv::drawContours(digital_output, contour_vec, 0, color_to_draw, cv::FILLED);
    }

    // Çizgileri de scale et
    cv::Mat template_lines = template_processor_->getTemplateLines();
    cv::Mat scaled_lines;
    cv::resize(template_lines, scaled_lines, warped_size, 0, 0, cv::INTER_NEAREST);
    digital_output.setTo(cv::Scalar(0, 0, 0), scaled_lines);

    return digital_output;
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
        cv::Mat digital = generateDigitalOutput(warped);

        // Artık ikisi de aynı boyutta!
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
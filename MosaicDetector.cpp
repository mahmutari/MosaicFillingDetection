#include "MosaicDetector.h"
#include <fstream>  // Dosya yazma için
#include <map>

// ==================== COLOR HISTORY ====================
ColorHistory::ColorHistory(size_t max_history) : max_history_(max_history) {}

void ColorHistory::addColor(const cv::Scalar& color) {
    recent_colors_.push_back(color);
    if (recent_colors_.size() > max_history_) {
        recent_colors_.erase(recent_colors_.begin());
    }
}

cv::Scalar ColorHistory::getStableColor() const {
    if (recent_colors_.empty()) {
        return cv::Scalar(255, 255, 255);
    }

    std::map<int, int> color_votes;
    for (const auto& color : recent_colors_) {
        int color_id = static_cast<int>(color[0]) +
            (static_cast<int>(color[1]) << 8) +
            (static_cast<int>(color[2]) << 16);
        color_votes[color_id]++;
    }

    int max_votes = 0;
    int winning_color_id = 0;
    for (const auto& pair : color_votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            winning_color_id = pair.first;
        }
    }

    int b = winning_color_id & 0xFF;
    int g = (winning_color_id >> 8) & 0xFF;
    int r = (winning_color_id >> 16) & 0xFF;

    return cv::Scalar(b, g, r);
}

void ColorHistory::clear() {
    recent_colors_.clear();
}

// ==================== COLOR DETECTOR ====================
ColorDetector::ColorDetector(int min_value, int max_value,
    int min_saturation, int threshold_divisor)
    : min_value_(min_value), max_value_(max_value),
    min_saturation_(min_saturation),
    color_threshold_divisor_(threshold_divisor) {
}

bool ColorDetector::isRed(int h, int r, int g, int b) const {
    // Sadece GERÇEKten kýrmýzý olan pikseller
    bool is_hsv_red = ((h >= 0 && h <= 10) || (h >= 170 && h <= 180));
    bool is_bgr_red = (r > 150) && (r > g * 1.5) && (r > b * 1.5);
    return is_hsv_red && is_bgr_red;
}

bool ColorDetector::isOrange(int h, int r, int g, int b) const {
    return (h >= 19 && h <= 30) && (r > g && r > b);
}

bool ColorDetector::isYellow(int h, int r, int g) const {
    return (h >= 31 && h <= 50) && (r > 80 && g > 80);
}

bool ColorDetector::isGreen(int h, int g, int r, int b) const {
    return (h >= 51 && h <= 90) && (g > r + 15 && g > b + 15);
}

bool ColorDetector::isBlue(int h, int b, int r, int g) const {
    return (h >= 91 && h <= 135) && (b > r + 20 && b > g + 15);
}

bool ColorDetector::isPurple(int h, int r, int g, int b) const {
    return (h >= 136 && h <= 161) && (r > g && b > g);
}

cv::Scalar ColorDetector::detectDominantColor(const cv::Mat& roi_bgr,
    const cv::Mat& roi_hsv,
    const cv::Mat& mask) const {
    if (roi_bgr.empty() || roi_hsv.empty()) {
        return cv::Scalar(255, 255, 255);
    }

    int red_count = 0, blue_count = 0, green_count = 0;
    int yellow_count = 0, purple_count = 0, orange_count = 0;
    int total_colored_pixels = 0;

    for (int y = 0; y < roi_hsv.rows; y++) {
        for (int x = 0; x < roi_hsv.cols; x++) {
            if (!mask.empty() && mask.at<uchar>(y, x) == 0) continue;

            cv::Vec3b hsv_pixel = roi_hsv.at<cv::Vec3b>(y, x);
            cv::Vec3b bgr_pixel = roi_bgr.at<cv::Vec3b>(y, x);

            int h = hsv_pixel[0], s = hsv_pixel[1], v = hsv_pixel[2];
            int b = bgr_pixel[0], g = bgr_pixel[1], r = bgr_pixel[2];

            if (v < min_value_ || v > max_value_ || s < min_saturation_) continue;

            total_colored_pixels++;

            if (isRed(h, r, g, b)) red_count++;
            else if (isOrange(h, r, g, b)) orange_count++;
            else if (isYellow(h, r, g)) yellow_count++;
            else if (isGreen(h, g, r, b)) green_count++;
            else if (isBlue(h, b, r, g)) blue_count++;
            else if (isPurple(h, r, g, b)) purple_count++;
        }
    }

    // --- YENÝ GÜNCELLENMÝÞ MANTIK (GÜRÜLTÜ FÝLTRELÝ) ---

    // Bir rengin "baskýn" sayýlmasý için o bölgedeki piksellerin 
    // en az %20'sini oluþturmasý gerekir.
    // Bu, gürültünün yanlýþlýkla renk olarak algýlanmasýný engeller.
    int threshold = std::max(10, total_colored_pixels / 5); // %20 eþiði

    int max_count = 0; // max_count'u 0'dan baþlatýyoruz
    cv::Scalar result_color = cv::Scalar(255, 255, 255); // Varsayýlan BEYAZ

    // Bir rengin kazanmasý için hem eþiði (threshold) 
    // hem de o ana kadarki maksimumu (max_count) geçmesi gerekir.
    if (red_count > threshold && red_count > max_count) {
        max_count = red_count;
        result_color = cv::Scalar(0, 0, 255);
    }
    if (blue_count > threshold && blue_count > max_count) {
        max_count = blue_count;
        result_color = cv::Scalar(255, 0, 0);
    }
    if (green_count > threshold && green_count > max_count) {
        max_count = green_count;
        result_color = cv::Scalar(0, 255, 0);
    }
    if (yellow_count > threshold && yellow_count > max_count) {
        max_count = yellow_count;
        result_color = cv::Scalar(0, 255, 255);
    }
    if (purple_count > threshold && purple_count > max_count) {
        max_count = purple_count;
        result_color = cv::Scalar(255, 0, 255); // Not: Bu BGR kodu Pembe/Magenta'dýr
    }
    if (orange_count > threshold && orange_count > max_count) {
        max_count = orange_count; // (Orijinal kodda bu satýr eksikti)
        result_color = cv::Scalar(0, 165, 255);
    }

    // Eðer hiçbir renk %20 eþiðini geçemezse (max_count = 0 kalýrsa),
    // fonksiyon varsayýlan "beyaz" rengi döndürür.
    return result_color;
}

// ==================== MARKER DETECTOR ====================
MarkerDetector::MarkerDetector(int target_id,
    const cv::aruco::Dictionary& dictionary,
    const cv::aruco::DetectorParameters& params)
    : target_marker_id_(target_id), detector_(dictionary, params) {
}

cv::Point2f MarkerDetector::getMarkerCenter(
    const std::vector<cv::Point2f>& corners) const {
    float x = 0, y = 0;
    for (const auto& corner : corners) { x += corner.x; y += corner.y; }
    return cv::Point2f(x / 4.0f, y / 4.0f);
}

bool MarkerDetector::detectMarkers(
    const cv::Mat& frame,
    std::vector<std::vector<cv::Point2f>>& target_corners,
    std::vector<int>& all_ids,
    std::vector<std::vector<cv::Point2f>>& all_corners) {

    detector_.detectMarkers(frame, all_corners, all_ids);
    target_corners.clear();
    if (!all_ids.empty()) {
        for (size_t i = 0; i < all_ids.size(); ++i) {
            if (all_ids[i] == target_marker_id_) {
                target_corners.push_back(all_corners[i]);
            }
        }
    }
    return target_corners.size() == 4;
}

std::vector<cv::Point2f> MarkerDetector::getOrderedCorners(
    const std::vector<std::vector<cv::Point2f>>& target_corners) const {

    std::vector<std::pair<int, cv::Point2f>> marker_centers;
    for (size_t i = 0; i < target_corners.size(); ++i) {
        marker_centers.push_back({ static_cast<int>(i), getMarkerCenter(target_corners[i]) });
    }

    auto temp_centers = marker_centers;

    std::sort(temp_centers.begin(), temp_centers.end(),
        [](const auto& a, const auto& b) { return (a.second.x + a.second.y) < (b.second.x + b.second.y); });
    int tl_idx = temp_centers[0].first;
    int br_idx = temp_centers[3].first;

    std::sort(temp_centers.begin(), temp_centers.end(),
        [](const auto& a, const auto& b) { return (a.second.x - a.second.y) > (b.second.x - b.second.y); });
    int tr_idx = temp_centers[0].first;
    int bl_idx = temp_centers[3].first;

    std::vector<cv::Point2f> ordered_corners(4);
    ordered_corners[0] = target_corners[tl_idx][0];
    ordered_corners[1] = target_corners[tr_idx][1];
    ordered_corners[2] = target_corners[br_idx][2];
    ordered_corners[3] = target_corners[bl_idx][3];

    return ordered_corners;
}

int MarkerDetector::getTargetId() const { return target_marker_id_; }

// ==================== TEMPLATE PROCESSOR ====================
TemplateProcessor::TemplateProcessor(const std::string& template_path) {
    template_image_ = cv::imread(template_path);
    if (template_image_.empty()) {
        throw std::runtime_error("Failed to load template: " + template_path);
    }
    output_size_ = template_image_.size();
    extractContoursAndLines();
}

void TemplateProcessor::extractContoursAndLines() {
    cv::Mat gray;
    cv::cvtColor(template_image_, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, template_lines_, 200, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(template_lines_, template_lines_, kernel);

    cv::Mat inverse;
    cv::bitwise_not(template_lines_, inverse);

    std::vector<std::vector<cv::Point>> all_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(inverse, all_contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    contours_.clear();
    for (const auto& contour : all_contours) {
        if (cv::contourArea(contour) > 200) {
            contours_.push_back(contour);
        }
    }
    std::cout << "Found " << contours_.size() << " mosaic pieces." << std::endl;
}

const cv::Mat& TemplateProcessor::getTemplateLines() const { return template_lines_; }
const std::vector<std::vector<cv::Point>>& TemplateProcessor::getContours() const { return contours_; }
cv::Size TemplateProcessor::getOutputSize() const { return output_size_; }
int TemplateProcessor::getWidth() const { return output_size_.width; }
int TemplateProcessor::getHeight() const { return output_size_.height; }

// ==================== MOSAIC DETECTOR ====================
MosaicDetector::MosaicDetector(const std::string& template_path,
    int target_marker_id,
    int camera_index)
    : is_running_(false) {

    template_processor_ = std::make_unique<TemplateProcessor>(template_path);

    // ======== GÜNCELLEME 1: BU SATIRIN YORUMUNU KALDIRDIK ========
    color_histories_.resize(template_processor_->getContours().size());

    color_detector_ = std::make_unique<ColorDetector>();

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.cornerRefinementWinSize = 7;
    params.cornerRefinementMaxIterations = 50;
    params.cornerRefinementMinAccuracy = 0.01;

    marker_detector_ = std::make_unique<MarkerDetector>(target_marker_id, dictionary, params);

    camera_.open(camera_index);
    if (!camera_.isOpened()) throw std::runtime_error("Failed to open camera!");
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    initializeWindows();
}

MosaicDetector::~MosaicDetector() { stop(); }

void MosaicDetector::initializeWindows() {
    cv::namedWindow("Live Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Bird's-Eye View", cv::WINDOW_NORMAL);
    cv::namedWindow("Digital Output", cv::WINDOW_NORMAL);
}

cv::Mat MosaicDetector::applyPerspectiveTransform(const cv::Mat& frame, const std::vector<cv::Point2f>& src_points) {
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(template_processor_->getWidth() - 1), 0),
        cv::Point2f(static_cast<float>(template_processor_->getWidth() - 1),
                   static_cast<float>(template_processor_->getHeight() - 1)),
        cv::Point2f(0, static_cast<float>(template_processor_->getHeight() - 1))
    };

    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(frame, warped, perspective_matrix, template_processor_->getOutputSize(),
        cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // ======== GÜNCELLEME 2: BLUR ÝÞLEMÝNÝ YORUM SATIRI YAPTIK (NETLÝK ÝÇÝN) ========
    // cv::GaussianBlur(warped, warped, cv::Size(3, 3), 0);

    return warped;
}

// ======== GÜNCELLEME 3: BU FONKSÝYONUN TAMAMINI YENÝ "DOLU" MANTIKLA DEÐÝÞTÝRDÝK ========
// ======== BU, YENÝ "UNUTMA" MANTIÐINA SAHÝP GÜNCEL KODDUR ========
cv::Mat MosaicDetector::generateDigitalOutput(const cv::Mat& warped_frame) {
    // 1. Kuþbakýþý görüntüyü HSV'ye çevir
    cv::Mat hsv_warped;
    cv::cvtColor(warped_frame, hsv_warped, cv::COLOR_BGR2HSV);

    // 2. Boþ, beyaz bir dijital çýktý görüntüsü oluþtur
    cv::Mat digital_output(template_processor_->getOutputSize(), CV_8UC3, cv::Scalar(255, 255, 255));

    // 3. Þablondan tüm mozaik parçalarýný (konturlarý) al
    const auto& contours = template_processor_->getContours();

    // 4. Her bir mozaik parçasý (kontur) üzerinde tek tek dön
    for (size_t i = 0; i < contours.size(); ++i) {

        // 5. Sadece bu parçayý seçen bir maske oluþtur
        cv::Mat mask = cv::Mat::zeros(warped_frame.size(), CV_8U);
        cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

        // 6. Bu bölgedeki baskýn rengi bul
        cv::Scalar dominant_color = color_detector_->detectDominantColor(warped_frame, hsv_warped, mask);

        // 7. ---- YENÝ UNUTMA MANTIÐI BURADA ----
        cv::Scalar color_to_draw;

        // detectDominantColor'ýn varsayýlan "renk yok" çýktýsý beyazdýr (255, 255, 255)
        bool is_white = (dominant_color[0] == 255 && dominant_color[1] == 255 && dominant_color[2] == 255);

        if (is_white) {
            // Eðer bölge beyazsa:
            // a. Renk geçmiþini (hafýzayý) temizle
            color_histories_[i].clear();
            // b. Çizilecek rengi beyaz olarak ayarla
            color_to_draw = cv::Scalar(255, 255, 255);
        }
        else {
            // Eðer bölge beyaz DEÐÝLSE (örn: maviyse):
            // a. Rengi stabilize etmek için geçmiþe ekle
            color_histories_[i].addColor(dominant_color);
            // b. Çizilecek stabil rengi geçmiþten al
            color_to_draw = color_histories_[i].getStableColor();
        }

        // 9. Bütün bölgeyi (konturu) bu renkle doldur
        cv::drawContours(digital_output, contours, static_cast<int>(i), color_to_draw, cv::FILLED);
    }

    // 10. (En son) Þablonun siyah çizgilerini üste ekle
    digital_output.setTo(cv::Scalar(0, 0, 0), template_processor_->getTemplateLines());

    return digital_output;
}// ======== GÜNCELLEME 3 BÝTTÝ ========


void MosaicDetector::run() {
    is_running_ = true;
    std::cout << "Starting Mosaic Detector... Press 'q' to quit." << std::endl;

    while (is_running_) {
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) break;

        processFrame(frame);

        if (static_cast<char>(cv::waitKey(1)) == 'q') break;
    }
    stop();
}

void MosaicDetector::processFrame(cv::Mat& frame) {
    std::vector<int> all_ids;
    std::vector<std::vector<cv::Point2f>> all_corners, target_corners;

    bool markers_found = marker_detector_->detectMarkers(frame, target_corners, all_ids, all_corners);

    cv::Mat display_frame = frame.clone();
    if (!all_ids.empty()) {
        cv::aruco::drawDetectedMarkers(display_frame, all_corners, all_ids);
    }

    if (markers_found) {
        auto ordered_corners = marker_detector_->getOrderedCorners(target_corners);

        for (size_t i = 0; i < ordered_corners.size(); i++) {
            cv::circle(display_frame, ordered_corners[i], 10, cv::Scalar(0, 255, 255), -1);
        }

        cv::Mat warped_frame = applyPerspectiveTransform(frame, ordered_corners);
        cv::imshow("Bird's-Eye View", warped_frame);
        cv::imshow("Digital Output", generateDigitalOutput(warped_frame));
    }

    cv::imshow("Live Video", display_frame);
}

void MosaicDetector::stop() {
    is_running_ = false;
    if (camera_.isOpened()) camera_.release();
    cv::destroyAllWindows();
}
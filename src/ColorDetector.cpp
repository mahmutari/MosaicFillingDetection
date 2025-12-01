#include "ColorDetector.h"
#include <algorithm>

ColorDetector::ColorDetector(int min_val, int max_val, int min_sat)
    : min_value_(min_val), max_value_(max_val), min_saturation_(min_sat) {
}

// ===================== RENK TESPİT FONKSİYONLARI =====================

// KIRMIZI: Hue 0-10 veya 170-180 (kırmızı HSV'de iki uçta)
bool ColorDetector::isRed(int h, int r, int g, int b) const {
    bool is_hsv_red = ((h >= 0 && h <= 10) || (h >= 170 && h <= 180));
    bool is_bgr_red = (r > 100) && (r > g * 1.3) && (r > b * 1.3);
    return is_hsv_red && is_bgr_red;
}

// TURUNCU: Hue 11-25
bool ColorDetector::isOrange(int h, int r, int g, int b) const {
    bool is_hsv_orange = (h >= 11 && h <= 25);
    bool is_bgr_orange = (r > 120) && (g > 50) && (g < r) && (b < g);
    return is_hsv_orange && is_bgr_orange;
}

// SARI: Hue 26-34
bool ColorDetector::isYellow(int h, int r, int g) const {
    bool is_hsv_yellow = (h >= 26 && h <= 34);
    bool is_bgr_yellow = (r > 120) && (g > 120) && (abs(r - g) < 60);
    return is_hsv_yellow && is_bgr_yellow;
}

// YEŞİL: Hue 35-85
bool ColorDetector::isGreen(int h, int g, int r, int b) const {
    bool is_hsv_green = (h >= 35 && h <= 85);
    bool is_bgr_green = (g >= 60) && (g > r * 1.05) && (g > b * 1.05);
    return is_hsv_green && is_bgr_green;
}

// MAVİ: Hue 90-120 (daraltıldı)
// Mavi için R kanalı düşük olmalı
bool ColorDetector::isBlue(int h, int b, int r, int g) const {
    bool is_hsv_blue = (h >= 90 && h <= 120);
    bool is_bgr_blue = (b >= 80) && (b > r * 1.2) && (b > g * 1.1) && (r < 120);
    return is_hsv_blue && is_bgr_blue;
}

// MOR/PEMBE: Hue 121-170 (genişletildi)
// Mor için hem R hem B yüksek, G düşük
bool ColorDetector::isPurple(int h, int r, int g, int b) const {
    bool is_hsv_purple = (h >= 121 && h <= 170);
    bool is_bgr_purple = (r > 60) && (b > 60) && (r > g) && (b > g);

    // Ek kontrol: R ve B birbirine yakın olmalı (mor karakteristiği)
    bool rb_balanced = (abs(r - b) < 100);

    return is_hsv_purple && is_bgr_purple && rb_balanced;
}

// ===================== YARDIMCI FONKSİYONLAR =====================

std::string ColorDetector::getColorName(const cv::Scalar& color) const {
    int b = static_cast<int>(color[0]);
    int g = static_cast<int>(color[1]);
    int r = static_cast<int>(color[2]);

    if (r == 255 && g == 255 && b == 255) return "White";
    if (r == 255 && g == 0 && b == 0) return "Red";
    if (r == 255 && g == 165 && b == 0) return "Orange";
    if (r == 255 && g == 255 && b == 0) return "Yellow";
    if (r == 0 && g == 255 && b == 0) return "Green";
    if (r == 0 && g == 0 && b == 255) return "Blue";
    if (r == 255 && g == 0 && b == 255) return "Purple";

    return "Unknown";
}

// ===================== ANA TESPİT FONKSİYONU =====================

ColorDetectionResult ColorDetector::detectColorWithRatio(const cv::Mat& roi_bgr,
    const cv::Mat& roi_hsv,
    const cv::Mat& mask) const {

    ColorDetectionResult result;
    result.fill_ratio = 0.0f;

    int red = 0, orange = 0, yellow = 0, green = 0, blue = 0, purple = 0;
    int total_valid_pixels = 0;
    int total_non_white_pixels = 0;

    for (int y = 0; y < roi_hsv.rows; y++) {
        for (int x = 0; x < roi_hsv.cols; x++) {
            if (mask.at<uchar>(y, x) == 0) continue;

            total_valid_pixels++;

            cv::Vec3b hsv = roi_hsv.at<cv::Vec3b>(y, x);
            cv::Vec3b bgr = roi_bgr.at<cv::Vec3b>(y, x);

            int h = hsv[0], s = hsv[1], v = hsv[2];
            int b = bgr[0], g = bgr[1], r = bgr[2];

            // Beyaz/gri tespiti
            bool is_white_or_gray = (s < 35) || (v > 230 && s < 50);

            // Çok koyu pikseller
            bool is_too_dark = (v < 40);

            if (is_white_or_gray || is_too_dark) {
                continue;
            }

            total_non_white_pixels++;

            // Renk kategorilerini kontrol et
            // ÖNCELİK SIRASI ÖNEMLİ: Mor, maviden önce kontrol edilmeli
            if (isRed(h, r, g, b)) red++;
            else if (isOrange(h, r, g, b)) orange++;
            else if (isYellow(h, r, g)) yellow++;
            else if (isGreen(h, g, r, b)) green++;
            else if (isPurple(h, r, g, b)) purple++;  // Mor önce
            else if (isBlue(h, b, r, g)) blue++;      // Mavi sonra
        }
    }

    // Dominant rengi bul
    int threshold = std::max(3, total_non_white_pixels / 15);

    int max_count = 0;
    cv::Scalar dominant_color(255, 255, 255);

    if (red > threshold && red > max_count) {
        max_count = red;
        dominant_color = cv::Scalar(0, 0, 255);      // BGR: Kırmızı
    }
    if (orange > threshold && orange > max_count) {
        max_count = orange;
        dominant_color = cv::Scalar(0, 165, 255);    // BGR: Turuncu
    }
    if (yellow > threshold && yellow > max_count) {
        max_count = yellow;
        dominant_color = cv::Scalar(0, 255, 255);    // BGR: Sarı
    }
    if (green > threshold && green > max_count) {
        max_count = green;
        dominant_color = cv::Scalar(0, 255, 0);      // BGR: Yeşil
    }
    if (blue > threshold && blue > max_count) {
        max_count = blue;
        dominant_color = cv::Scalar(255, 0, 0);      // BGR: Mavi
    }
    if (purple > threshold && purple > max_count) {
        max_count = purple;
        dominant_color = cv::Scalar(255, 0, 255);    // BGR: Mor
    }

    result.color = dominant_color;
    result.color_name = getColorName(dominant_color);

    if (total_valid_pixels > 0) {
        result.fill_ratio = static_cast<float>(total_non_white_pixels) / static_cast<float>(total_valid_pixels);
    }

    return result;
}

cv::Scalar ColorDetector::detectDominantColor(const cv::Mat& roi_bgr,
    const cv::Mat& roi_hsv,
    const cv::Mat& mask) const {
    ColorDetectionResult result = detectColorWithRatio(roi_bgr, roi_hsv, mask);
    return result.color;
}
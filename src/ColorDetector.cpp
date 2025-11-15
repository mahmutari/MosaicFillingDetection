#include "ColorDetector.h"
#include <algorithm> // std::max

ColorDetector::ColorDetector(int min_val, int max_val, int min_sat)
    : min_value_(min_val), max_value_(max_val), min_saturation_(min_sat) {
}

bool ColorDetector::isRed(int h, int r, int g, int b) const {
    bool is_hsv_red = ((h >= 0 && h <= 10) || (h >= 170 && h <= 180));
    bool is_bgr_red = (r > 140) && (r > g * 1.5) && (r > b * 1.5);
    return is_hsv_red && is_bgr_red;
}
bool ColorDetector::isOrange(int h, int r, int g, int b) const {
    return (h >= 11 && h <= 25) && (r > 150) && (g > 60) && (r > g * 1.2);
}
bool ColorDetector::isYellow(int h, int r, int g) const {
    return (h >= 26 && h <= 40) && (r > 150) && (g > 150);
}
bool ColorDetector::isGreen(int h, int g, int r, int b) const {
    return (h >= 45 && h <= 85) && (g > 130) && (g > r * 1.3);
}
bool ColorDetector::isBlue(int h, int b, int r, int g) const {
    return (h >= 100 && h <= 130) && (b > 130) && (b > r * 1.2);
}
bool ColorDetector::isPurple(int h, int r, int g, int b) const {
    return (h >= 135 && h <= 160) && (r > 110) && (b > 110);
}

cv::Scalar ColorDetector::detectDominantColor(const cv::Mat& roi_bgr,
    const cv::Mat& roi_hsv,
    const cv::Mat& mask) const {

    int red = 0, orange = 0, yellow = 0, green = 0, blue = 0, purple = 0;
    int total_colored_pixels = 0;

    for (int y = 0; y < roi_hsv.rows; y++) {
        for (int x = 0; x < roi_hsv.cols; x++) {
            if (mask.at<uchar>(y, x) == 0) continue;

            cv::Vec3b hsv = roi_hsv.at<cv::Vec3b>(y, x);
            cv::Vec3b bgr = roi_bgr.at<cv::Vec3b>(y, x);
            int h = hsv[0], s = hsv[1], v = hsv[2];
            int b = bgr[0], g = bgr[1], r = bgr[2];

            if (v < min_value_ || v > max_value_ || s < min_saturation_) continue;
            total_colored_pixels++;

            if (isRed(h, r, g, b)) red++;
            else if (isOrange(h, r, g, b)) orange++;
            else if (isYellow(h, r, g)) yellow++;
            else if (isGreen(h, g, r, b)) green++;
            else if (isBlue(h, b, r, g)) blue++;
            else if (isPurple(h, r, g, b)) purple++;
        }
    }

    // Gürültü filtresi: Bir rengin kazanmasý için piksellerin en az %20'sini oluþturmasý gerekir.
    int threshold = std::max(10, total_colored_pixels / 5);
    int max_count = 0;
    cv::Scalar result_color(255, 255, 255); // Varsayýlan beyaz

    if (red > threshold && red > max_count) { max_count = red; result_color = cv::Scalar(0, 0, 255); }
    if (orange > threshold && orange > max_count) { max_count = orange; result_color = cv::Scalar(0, 165, 255); }
    if (yellow > threshold && yellow > max_count) { max_count = yellow; result_color = cv::Scalar(0, 255, 255); }
    if (green > threshold && green > max_count) { max_count = green; result_color = cv::Scalar(0, 255, 0); }
    if (blue > threshold && blue > max_count) { max_count = blue; result_color = cv::Scalar(255, 0, 0); }
    if (purple > threshold && purple > max_count) { max_count = purple; result_color = cv::Scalar(255, 0, 255); }

    return result_color;
}
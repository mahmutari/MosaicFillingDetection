#pragma once
#include <opencv2/opencv.hpp>

struct ColorDetectionResult {
    cv::Scalar color;           // Tespit edilen renk (BGR)
    std::string color_name;     // Renk adý ("Red", "Blue", vs.)
    float fill_ratio;           // Doluluk oraný (0.0 - 1.0)
};

class ColorDetector {
private:
    int min_value_;
    int max_value_;
    int min_saturation_;

    bool isRed(int h, int r, int g, int b) const;
    bool isOrange(int h, int r, int g, int b) const;
    bool isYellow(int h, int r, int g) const;
    bool isGreen(int h, int g, int r, int b) const;
    bool isBlue(int h, int b, int r, int g) const;
    bool isPurple(int h, int r, int g, int b) const;

    std::string getColorName(const cv::Scalar& color) const;

public:
    ColorDetector(int min_val = 40, int max_val = 240, int min_sat = 50);

    // Eski fonksiyon - geriye uyumluluk
    cv::Scalar detectDominantColor(const cv::Mat& roi_bgr,
        const cv::Mat& roi_hsv,
        const cv::Mat& mask) const;

    // Yeni fonksiyon - ratio bilgisi ile
    ColorDetectionResult detectColorWithRatio(const cv::Mat& roi_bgr,
        const cv::Mat& roi_hsv,
        const cv::Mat& mask) const;
};
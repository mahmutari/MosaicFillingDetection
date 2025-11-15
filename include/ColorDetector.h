#pragma once
#include <opencv2/opencv.hpp>

class ColorDetector {
private:
    int min_value_;
    int max_value_;
    int min_saturation_;

    // Renk tespiti için özel yardýmcý fonksiyonlar
    bool isRed(int h, int r, int g, int b) const;
    bool isOrange(int h, int r, int g, int b) const;
    bool isYellow(int h, int r, int g) const;
    bool isGreen(int h, int g, int r, int b) const;
    bool isBlue(int h, int b, int r, int g) const;
    bool isPurple(int h, int r, int g, int b) const;

public:
    ColorDetector(int min_val = 60, int max_val = 240, int min_sat = 40);

    cv::Scalar detectDominantColor(const cv::Mat& roi_bgr,
        const cv::Mat& roi_hsv,
        const cv::Mat& mask) const;
};
#pragma once
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

class ColorHistory {
private:
    std::vector<cv::Scalar> recent_colors_;
    size_t max_history_;
public:
    explicit ColorHistory(size_t max_history = 7);
    void addColor(const cv::Scalar& color);
    cv::Scalar getStableColor() const;
    void clear();
};
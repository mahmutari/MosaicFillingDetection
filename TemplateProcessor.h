#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class TemplateProcessor {
private:
    cv::Mat template_image_;
    cv::Mat template_lines_;
    std::vector<std::vector<cv::Point>> contours_;
    cv::Size output_size_;

    void extractContoursAndLines();

public:
    explicit TemplateProcessor(const std::string& template_path);

    const cv::Mat& getTemplateLines() const;
    const std::vector<std::vector<cv::Point>>& getContours() const;
    cv::Size getOutputSize() const;
};
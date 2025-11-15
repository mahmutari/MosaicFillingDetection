#include "TemplateProcessor.h"
#include <stdexcept>
#include <iostream>

TemplateProcessor::TemplateProcessor(const std::string& template_path) {
    template_image_ = cv::imread(template_path);
    if (template_image_.empty()) {
        throw std::runtime_error("Failed to load template: " + template_path);
    }
    output_size_ = template_image_.size();
    extractContoursAndLines();
}

// Konturlarý *bir kez* burada bulup saklýyoruz (verimli yöntem)
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
        if (cv::contourArea(contour) > 200 && cv::contourArea(contour) < (output_size_.area() * 0.2)) {
            contours_.push_back(contour);
        }
    }
    std::cout << "Found " << contours_.size() << " mosaic pieces." << std::endl;
}

const cv::Mat& TemplateProcessor::getTemplateLines() const { return template_lines_; }
const std::vector<std::vector<cv::Point>>& TemplateProcessor::getContours() const { return contours_; }
cv::Size TemplateProcessor::getOutputSize() const { return output_size_; }
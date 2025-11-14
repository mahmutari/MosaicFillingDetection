#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Modüler sýnýflarýmýzý dahil ediyoruz

#include "build/ColorHistory.h"
#include "build/ColorDetector.h"
#include "build/MarkerDetector.h"
#include "build/TemplateProcessor.h"

class MosaicDetector {
private:
    // Alt sistemler için iþaretçiler
    std::unique_ptr<TemplateProcessor> template_processor_;
    std::unique_ptr<MarkerDetector> marker_detector_;
    std::unique_ptr<ColorDetector> color_detector_;

    // Her kontur için bir renk geçmiþi
    std::vector<ColorHistory> color_histories_;

    cv::VideoCapture camera_;
    bool is_running_;

    // Private yardýmcý fonksiyonlar
    void initializeWindows();
    cv::Mat applyPerspectiveTransform(const cv::Mat& frame,
        const std::vector<cv::Point2f>& src_points);
    cv::Mat generateDigitalOutput(const cv::Mat& warped_frame);

public:
    MosaicDetector(const std::string& template_path, int target_marker_id,
        int camera_index = 0);
    ~MosaicDetector();

    void run();
    void processFrame(cv::Mat& frame); // run() tarafýndan çaðrýlýr
    void stop();
};
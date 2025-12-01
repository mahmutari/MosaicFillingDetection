#pragma once
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MarkerDetector.h"
#include "TemplateProcessor.h"
#include "ColorDetector.h"
#include "ColorHistory.h"

struct PatchInfo {
    int patch_id;
    std::string color_name;
    float fill_ratio;
    cv::Point centroid;
};

class MosaicDetector {
private:
    std::unique_ptr<MarkerDetector> marker_detector_;
    std::unique_ptr<TemplateProcessor> template_processor_;
    std::unique_ptr<ColorDetector> color_detector_;
    std::vector<ColorHistory> color_histories_;
    std::vector<float> ratio_histories_;

    cv::VideoCapture camera_;
    bool is_running_;

    void initializeWindows();

    cv::Mat applyPerspectiveTransform(const cv::Mat& frame,
        const std::vector<cv::Point2f>& src_points);

    cv::Mat generateDigitalOutput(const cv::Mat& warped_frame,
        std::vector<PatchInfo>& patch_infos);

    void drawRatioInfo(cv::Mat& image, const std::vector<PatchInfo>& patch_infos);
    cv::Point calculateContourCentroid(const std::vector<cv::Point>& contour);

public:
    MosaicDetector(const std::string& template_path,
        int target_marker_id = 23,
        int camera_index = 0);

    ~MosaicDetector();

    void run();
    void processFrame(cv::Mat& frame);
    void stop();
};
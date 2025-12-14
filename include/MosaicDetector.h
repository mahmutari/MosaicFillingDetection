#pragma once
#include <memory>
#include <vector>
#include <string>
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
    std::unique_ptr<ColorDetector> color_detector_;

    // Çoklu template desteði
    std::vector<std::unique_ptr<TemplateProcessor>> template_processors_;
    std::vector<std::string> template_paths_;
    std::vector<std::string> template_names_;
    int current_template_index_;
    int detected_template_index_;
    int template_vote_count_;

    std::vector<std::vector<ColorHistory>> all_color_histories_;
    std::vector<std::vector<float>> all_ratio_histories_;

    cv::VideoCapture camera_;
    bool is_running_;

    // Rotasyon takibi
    int current_rotation_;
    int rotation_vote_count_ = 0;

    void initializeWindows();
    void switchTemplate(int index);
    void resetHistories(int index);

    cv::Mat applyPerspectiveTransform(const cv::Mat& frame,
        const std::vector<cv::Point2f>& src_points);

    cv::Mat generateDigitalOutput(const cv::Mat& warped_frame,
        std::vector<PatchInfo>& patch_infos);

    void drawRatioInfo(cv::Mat& image, const std::vector<PatchInfo>& patch_infos);
    cv::Point calculateContourCentroid(const std::vector<cv::Point>& contour);

    // Rotasyon fonksiyonlarý
    int detectRotation(const std::vector<std::vector<cv::Point2f>>& markers);
    cv::Mat rotateImage(const cv::Mat& image, int rotation);
    cv::Mat rotateImageInverse(const cv::Mat& image, int rotation);

    // Otomatik template algýlama
    int detectTemplate(const cv::Mat& warped_normalized);
    double calculateTemplateSimilarity(const cv::Mat& warped_lines, int template_index);

public:
    MosaicDetector(const std::vector<std::string>& template_paths,
        const std::vector<std::string>& template_names,
        int target_marker_id = 23,
        int camera_index = 0);

    ~MosaicDetector();

    void run();
    void processFrame(cv::Mat& frame);
    void stop();
};
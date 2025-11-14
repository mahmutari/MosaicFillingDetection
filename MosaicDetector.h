#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

// Forward declarations
class ColorHistory;
class ColorDetector;
class MarkerDetector;
class TemplateProcessor;

// ==================== COLOR HISTORY CLASS ====================
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

// ==================== COLOR DETECTOR CLASS ====================
class ColorDetector {
private:
    int min_value_;
    int max_value_;
    int min_saturation_;
    int color_threshold_divisor_;

    bool isRed(int h, int r, int g, int b) const;
    bool isOrange(int h, int r, int g, int b) const;
    bool isYellow(int h, int r, int g) const;
    bool isGreen(int h, int g, int r, int b) const;
    bool isBlue(int h, int b, int r, int g) const;
    bool isPurple(int h, int r, int g, int b) const;

public:
    ColorDetector(int min_value = 40, int max_value = 245,
        int min_saturation = 35, int threshold_divisor = 12);

    cv::Scalar detectDominantColor(const cv::Mat& roi_bgr,
        const cv::Mat& roi_hsv,
        const cv::Mat& mask) const;
};

// ==================== MARKER DETECTOR CLASS ====================
class MarkerDetector {
private:
    int target_marker_id_;
    cv::aruco::ArucoDetector detector_;

    cv::Point2f getMarkerCenter(const std::vector<cv::Point2f>& corners) const;

public:
    MarkerDetector(int target_id, const cv::aruco::Dictionary& dictionary,
        const cv::aruco::DetectorParameters& params);

    bool detectMarkers(const cv::Mat& frame,
        std::vector<std::vector<cv::Point2f>>& target_corners,
        std::vector<int>& all_ids,
        std::vector<std::vector<cv::Point2f>>& all_corners);

    std::vector<cv::Point2f> getOrderedCorners(
        const std::vector<std::vector<cv::Point2f>>& target_corners) const;

    int getTargetId() const;
};

// ==================== TEMPLATE PROCESSOR CLASS ====================
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
    int getWidth() const;
    int getHeight() const;
};

// ==================== MOSAIC DETECTOR CLASS ====================
class MosaicDetector {
private:
    std::unique_ptr<MarkerDetector> marker_detector_;
    std::unique_ptr<TemplateProcessor> template_processor_;
    std::unique_ptr<ColorDetector> color_detector_;
    std::vector<ColorHistory> color_histories_;

    std::map<int, ColorHistory> dynamic_color_histories_;

    cv::VideoCapture camera_;
    bool is_running_;

    void initializeWindows();
    cv::Mat applyPerspectiveTransform(const cv::Mat& frame,
        const std::vector<cv::Point2f>& src_points);
    cv::Mat generateDigitalOutput(const cv::Mat& warped_frame);

public:
    MosaicDetector(const std::string& template_path, int target_marker_id,
        int camera_index = 0);
    ~MosaicDetector();

    void run();
    void processFrame(cv::Mat& frame);
    void stop();
};
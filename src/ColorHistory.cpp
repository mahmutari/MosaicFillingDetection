#include "ColorHistory.h"

ColorHistory::ColorHistory(size_t max_history) : max_history_(max_history) {}

void ColorHistory::addColor(const cv::Scalar& color) {
    recent_colors_.push_back(color);
    if (recent_colors_.size() > max_history_) {
        recent_colors_.erase(recent_colors_.begin());
    }
}

cv::Scalar ColorHistory::getStableColor() const {
    if (recent_colors_.empty()) {
        return cv::Scalar(255, 255, 255);
    }

    std::map<int, int> color_votes;
    for (const auto& color : recent_colors_) {
        int color_id = static_cast<int>(color[0]) +
            (static_cast<int>(color[1]) << 8) +
            (static_cast<int>(color[2]) << 16);
        color_votes[color_id]++;
    }

    int max_votes = 0;
    int winning_color_id = 0;
    for (const auto& pair : color_votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            winning_color_id = pair.first;
        }
    }

    int b = winning_color_id & 0xFF;
    int g = (winning_color_id >> 8) & 0xFF;
    int r = (winning_color_id >> 16) & 0xFF;

    return cv::Scalar(b, g, r);
}

void ColorHistory::clear() {
    recent_colors_.clear();
}
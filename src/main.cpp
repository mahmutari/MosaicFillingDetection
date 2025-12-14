#include "MosaicDetector.h"
#include <vector>
#include <string>

int main() {
    try {
        // Template dosya yollarý
        std::vector<std::string> template_paths = {
            "C:/Users/pc/Desktop/MosaicProject/MosaicFillingDetection/mosaic.jpg",
            "C:/Users/pc/Desktop/MosaicProject/MosaicFillingDetection/mosaic_2.jpg"
        };

        // Template isimleri (ekranda gösterilecek)
        std::vector<std::string> template_names = {
            "Gunes (Sun)",
            "Ay (Moon)"
        };

        int marker_id = 23;
        int camera_index = 0;

        std::cout << "=== Mosaic Detection System ===" << std::endl;
        std::cout << "Loading templates..." << std::endl;

        MosaicDetector detector(template_paths, template_names, marker_id, camera_index);
        detector.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
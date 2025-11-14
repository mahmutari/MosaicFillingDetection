#include "MosaicDetector.h"

int main() {
    try {
        MosaicDetector detector("mosaic.jpg", 23, 0);
        detector.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
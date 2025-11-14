# Gerekli minimum CMake versiyonunu belirt
cmake_minimum_required(VERSION 3.10)

# Projenin adını tanımla
project(MosaicFillingDetection)

# C++ 17 standardını kullan
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg aracılığıyla OpenCV'yi ve ArUco modülünü bul
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc aruco)

# Çalıştırılabilir dosyayı ve tüm .cpp dosyalarını tanımla
add_executable(MosaicFillingDetectionApp 
    main.cpp
    MosaicDetector.cpp
    ColorHistory.cpp
    ColorDetector.cpp
    MarkerDetector.cpp
    TemplateProcessor.cpp
)

# Derleyiciye header dosyaları için ana klasöre de bakmasını söyle
target_include_directories(MosaicFillingDetectionApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# OpenCV kütüphanelerini çalıştırılabilir dosyamıza bağla
target_link_libraries(MosaicFillingDetectionApp PRIVATE ${OpenCV_LIBS})
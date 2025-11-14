# [cite_start]Gerekli minimum CMake versiyonunu belirt [cite: 1]
cmake_minimum_required(VERSION 3.10)

# Projenin adını tanımla
project(MosaicFillingDetection)

# Kodumuzun C++ 17 standardını kullanmasını isteyelim (iyi bir alışkanlık)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# [cite_start]vcpkg aracılığıyla OpenCV'yi ve ArUco modülünü bul [cite: 3]
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc aruco)

# Projemize çalıştırılabilir bir dosya ekle
# DÜZELTME: main.cpp YANINDA MosaicDetector.cpp'yi de ekledik
add_executable(MosaicFillingDetectionApp 
    main.cpp
    MosaicDetector.cpp
    MosaicDetector.h
)

# OpenCV kütüphanelerini çalıştırılabilir dosyamıza bağla
target_link_libraries(MosaicFillingDetectionApp PRIVATE ${OpenCV_LIBS})
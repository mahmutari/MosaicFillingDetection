cmake_minimum_required(VERSION 3.15)

project(MosaicCMake VERSION 1.0 LANGUAGES CXX)

# OpenCV bul
find_package(OpenCV REQUIRED)

# Kaynak ve header dosyalarını otomatik topla
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "include/*.h")

# Executable oluştur
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

# Include dizinleri
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# VS için filtreler
source_group("Source Files" FILES ${SOURCES})
source_group("Header Files" FILES ${HEADERS})

# C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.17)
project(ConvolutionalNN VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ищем все cpp и hpp/h файлы
file(GLOB_RECURSE CPP_SOURCES "source/*.cpp")
file(GLOB_RECURSE HEADER_SOURCES "include/*.hpp" "include/*.h")

# Собираем библиотеку
add_library(ConvolutionalNN_static STATIC ${CPP_SOURCES} ${HEADER_SOURCES})

# Указываем папку include для поиска заголовков
target_include_directories(ConvolutionalNN_static PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Добавляем папку с приложениями
add_subdirectory(apps)

# Установка заголовочных файлов
install(
        DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Установка библиотеки
install(
        TARGETS ConvolutionalNN_static
        ARCHIVE DESTINATION lib
)


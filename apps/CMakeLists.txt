# Создаём исполняемый файл
add_executable(ConvolutionalNN_demo main.cpp)

# Линкуем его с вашей статической библиотекой
target_link_libraries(ConvolutionalNN_demo PRIVATE ConvolutionalNN_static)

# Подключаем заголовки библиотеки
target_include_directories(ConvolutionalNN_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
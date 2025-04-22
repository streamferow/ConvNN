add_library(ConvolutionalNN_static STATIC
        Tensor/Tensor.cpp
        Layer.cpp
        Utils/Memory.cpp
        )

target_include_directories(ConvolutionalNN_static
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )

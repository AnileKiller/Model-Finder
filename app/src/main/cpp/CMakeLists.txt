# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
project("beauty-engine")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
add_library(
        beauty-engine
        SHARED
        native-lib.cpp
        BeautyMath.cpp
        SkinProcessor.cpp
        BlemishEngine.cpp
        FeatureProcessor.cpp)

# Searches for a specified prebuilt library and stores the path as a variable.
find_library(log-lib log)

# Specifies libraries CMake should link to your target library.
target_link_libraries(beauty-engine ${log-lib})

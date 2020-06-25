#!/usr/bin/env bash

## ubuntu16.04
build_linux_fn () 
{
    echo "linux x86-64"
    mkdir -p build-linux
    pushd build-linux
    cmake -DNCNN_BENCHMARK=OFF \
          -DNCNN_OPENMP=ON \
          -DNCNN_REQUANT=OFF \
          -DNCNN_IM2COL_SGEMM=ON \
          ..
    make -j3
    make install
    popd
}

## arm linux
build_arm_fn () 
{
    echo "linux armv7-a"
    mkdir -p build-arm
    pushd build-arm
    cmake -DNCNN_OPENMP=ON \
          -DNCNN_REQUANT=OFF \
          -DNCNN_BENCHMARK=OFF \
          -DNCNN_IM2COL_SGEMM=ON \
          -DCMAKE_TOOLCHAIN_FILE=../arm.toolchain.cmake ..
    make -j3
    make install
    popd  
}

## android armv7
build_android_fn ()
{
    echo "android armv7-a"
    mkdir -p build-android-armv7
    pushd build-android-armv7
    cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-21 \
          -DNCNN_OPENMP=ON \
          -DNCNN_REQUANT=ON \
          -DNCNN_BENCHMARK=OFF \
          -DNCNN_IM2COL_SGEMM=ON \
          ..   
    make -j3
    make install
    popd
}

build_android_aarch64_fn ()
{
    echo "android aarch64"
    mkdir -p build-android-aarch64
    pushd build-android-aarch64
    cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 \
          -DNCNN_OPENMP=ON \
          -DNCNN_BENCHMARK=ON \
          -DNCNN_REQUANT=OFF \
          -DNCNN_IM2COL_SGEMM=OFF \
          ..
    make -j3
    make install
    popd   
}

build_clean_fn ()
{
    echo "remove build file"
    rm -rf build-android-armv7
    rm -rf build-android-aarch64
    rm -rf build-linux
    rm -rf build-arm
    echo "remove pull file"
    rm -rf output
    echo "build clean done"
}

error_fn () 
{
    echo "unknown argument"
    echo "available targets: linux||android||android_64||arm"
}

if [ $# = 0 ]; then
    echo "error: target missing!"
    echo "available targets: linux||android||android_64||arm"
    echo "sample usage: ./build.sh linux"
else
    if [ $1 = "linux" ]; then
        build_linux_fn
    elif [ $1 = "android" ]; then
        build_android_fn
    elif [ $1 = "android64" ]; then
        build_android_aarch64_fn        
    elif [ $1 = "arm" ]; then
        build_arm_fn   
    elif [ $1 = "clean" ]; then
        build_clean_fn
    else
        error_fn
    fi
fi

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "net.h"
#include "cpu.h"
#include "benchmark.h"

struct Object
{
    cv::Point p;
    float prob;
};

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static int pose_net(const cv::Mat& src_img, const cv::Mat& raw_img, const char* param_path, const char* bin_path, const int loop_count)
{
    float scale_x = src_img.cols / 157.0;
    float scale_y = src_img.rows / 157.0;

    ncnn::Net net;

    net.load_param(param_path);
    net.load_model(bin_path);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();   

    ncnn::Mat in = ncnn::Mat::from_pixels(raw_img.data, ncnn::Mat::PIXEL_BGR2RGB, raw_img.cols, raw_img.rows);

    const float mean_vals[3] = {103.f, 116.f, 122.f};
    const float norm_vals[3] = {1.0/255.0,1.0/255.0,1.0/255.0};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;

    for (int i = 0; i < loop_count; i++)
    {
        double start = ncnn::get_current_time();
        ncnn::Extractor ex = net.create_extractor();

        ex.input("data", in);
        ex.extract("ConvNd_183", out);
        double end = ncnn::get_current_time();
        printf("iter cost %d/%d: %.8lfms\n", i, loop_count, end - start);  
    }    

    printf("%d %d %d\n", out.w, out.h, out.c);
    //const float *values = (float*)out.data;
    std::vector<Object> objects;
    objects.clear();
    for (int ic=0;ic<out.c;ic++)
    {
        float maxv=0;
        int maxx=0;
        int maxy=0;
        for(int ih=0;ih<out.h;ih++)
        {
            for (int iw=0;iw<out.w;iw++)
            {
                float val=out.channel(ic).row(ih)[iw];
                if (maxv<val)
                {
                    maxv=val;
                    maxx=iw;
                    maxy=ih;
                }
            }
        }
        printf("now value: %f(%d,%d)\n", maxv, maxx, maxy);
        Object object;
        object.p.x = maxx * scale_x * 4.0;
        object.p.y = maxy * scale_y * 4.0;
        object.prob = maxv;

        objects.push_back(object);
    }

#ifndef __ARM_NEON
    // draw pose point
    cv::Mat image = src_img.clone();
    
    for(auto object : objects)
    {
        cv::circle(image, object.p, 3, cv::Scalar(255, 0, 0), 2);
    }
    
    cv::imshow("result",image);
    cv::waitKey();
#endif    

    return 0;
}

int main(int argc, char** argv)
{   
    std::cout << "--- DeepGlint ncnn post demo --- " << __TIME__ << " " << __DATE__ << std::endl; 

    if (argc != 7)
    {
        fprintf(stderr, "Usage: %s [imagepath] [parampath] [binpath] [loop count] [num thread] [powersave] \n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* parampath = argv[2];
    const char* binpath = argv[3];
    const int loop_count = atoi(argv[4]);
    const int num_threads = atoi(argv[5]);
    const int powersave = atoi(argv[6]);   

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    ncnn::set_default_option(opt);
    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);     

    // cv::Mat m1;
    // cv::cvtColor(m, m1, cv::COLOR_BGR2RGB);
    // cv::Mat m2_f;
    // m1.convertTo(m2_f, CV_32FC3, 1/225.0);
    // std::vector<cv::Mat> m3_v;
    // cv::split(m2_f, m3_v);
    // m3_v[0] -= 0.406;
    // m3_v[1] -= 0.457;
    // m3_v[2] -= 0.480;
    // cv::Mat m4;
    // cv::merge(m3_v, m4);
    cv::Mat m5;
    cv::resize(m, m5, cv::Size(157,157));

    pose_net(m, m5, parampath, binpath, loop_count);
    
    return 0;
}

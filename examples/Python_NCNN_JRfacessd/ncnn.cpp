#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>
#include <iostream>

#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "ncnn.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

Net::Net()
{
    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    ncnn::set_default_option(opt);
    ncnn::set_cpu_powersave(1);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(opt.num_threads);
}
Net::~Net()
{
}
int Net::load_param(const char *paramPath)
{
    return net.load_param(paramPath);
}

int Net::load_model(const char *modelPath)
{
    return net.load_model(modelPath);
}

void Net::setInputBlobName(string name)
{
    inputBlobNmae = name;
}
void Net::setOutputBlobName(string name)
{
    outputBlobName = name;
}

struct MyRect
{
    float x;
    float y;
    float w;
    float h;
};
struct Bbox
{
    float confidence;
    MyRect frect;
    bool deleted;
};

int MyRectInit(float x, float y, float width, float height, MyRect *rect)
{
    if (width == 0 || height == 0)
    {
        rect->x = 0;
        rect->y = 0;
        rect->w = 0;
        rect->h = 0;
    }
    else
    {
        rect->x = x;
        rect->y = y;
        rect->w = width;
        rect->h = height;
    }

    return 0;
}

int MyRectIntersect(MyRect rect1, MyRect rect2, MyRect *pDstFRect)
{
    float x = rect1.x > rect2.x ? rect1.x : rect2.x;
    float y = rect1.y > rect2.y ? rect1.y : rect2.y;
    float dw = (rect1.x + rect1.w < rect2.x + rect2.w ? rect1.x + rect1.w : rect2.x + rect2.w) - x;
    float dh = (rect1.y + rect1.h < rect2.y + rect2.h ? rect1.y + rect1.h : rect2.y + rect2.h) - y;

    float w = dw > 0 ? dw : 0;
    float h = dh > 0 ? dh : 0;

    return MyRectInit(x, y, w, h, pDstFRect);
}

float MyRectIou(MyRect rect1, MyRect rect2)
{
    float iou_val = 0.0f;
    MyRect frect;
    MyRectIntersect(rect1, rect2, &frect);

    float f32SrcArea = rect1.w * rect1.h;
    float f32DstArea = rect2.w * rect2.h;
    float f32IouArea = frect.w * frect.h;
    if (f32IouArea > 0)
    {
        iou_val = (float)f32IouArea / (f32SrcArea + f32DstArea - f32IouArea);
    }

    return iou_val;
}

void nms(std::vector<Bbox> &p, float nms_thres)
{
    std::sort(p.begin(), p.end(), [](const Bbox &a, const Bbox &b) { return a.confidence > b.confidence; });

    for (int i = 0; i < (int)p.size(); ++i)
    {
        if (p[i].deleted)
            continue;
        for (int j = i + 1; j < (int)p.size(); ++j)
        {
            if (!p[j].deleted)
            {
                if (MyRectIou(p[i].frect, p[j].frect) > nms_thres)
                {
                    p[j].deleted = true;
                }
            }
        }
    }
}

std::vector<Bbox> process_single_output_detection(ncnn::Mat &feat_conf, ncnn::Mat &feat_loc, float nms_thres)
{
    std::vector<float> variance_list = {0.1, 0.1, 0.2, 0.2};

    std::vector<int> min_size_vec = {10, 15, 20, 25, 35, 40};
    std::vector<float> aspect_ratio_vec = {1.0, 0.8};
    int step = 0;

    float offset = 0.5;
    int aspect_ratio_num = aspect_ratio_vec.size();
    int anchor_number = min_size_vec.size() * aspect_ratio_num;

    std::vector<Bbox> vbbox;

    int cstep = feat_conf.cstep;

    const float *cls_cpu = (const float *)feat_conf.data;
    const float *reg_cpu = (const float *)feat_loc.data;
    int img_height = 216;
    int img_width = 384;
    float _conf_thres = 0.1;
    int cls_height = feat_conf.h;
    int cls_width = feat_conf.w;
    int reg_height = feat_loc.h;
    int reg_width = feat_loc.w;
    assert(cls_height == reg_height);
    assert(cls_width == reg_width);

    float step_w = step;
    float step_h = step;
    if (step == 0)
    {
        step_w = img_width * 1.0 / reg_width;
        step_h = img_height * 1.0 / reg_height;
    }

    float log_thres[anchor_number];
    for (int i = 0; i < anchor_number; ++i)
    {
        log_thres[i] = std::log(_conf_thres / (1.0 - _conf_thres));
    }

    float pred_w, pred_h, center_x, center_y, pred_x, pred_y, raw_pred_x1,
        raw_pred_y1, raw_pred_w, raw_pred_h, prior_center_x, prior_center_y;
    for (int j = 0; j < anchor_number; ++j)
    {
        float aspect_ratio = aspect_ratio_vec[j % aspect_ratio_num];
        float prior_w = min_size_vec[j / aspect_ratio_num] * sqrt(aspect_ratio);
        float prior_h = min_size_vec[j / aspect_ratio_num] / sqrt(aspect_ratio);

        for (int y_index = 0; y_index < cls_height; y_index++)
        {
            for (int x_index = 0; x_index < cls_width; x_index++)
            {
                float x0 = cls_cpu[2 * j * cstep + y_index * cls_width + x_index];
                float x1 = cls_cpu[(2 * j + 1) * cstep + y_index * cls_width + x_index];
                if (x1 - x0 > log_thres[j])
                {
                    raw_pred_x1 = reg_cpu[j * 4 * cstep + y_index * reg_width + x_index];
                    raw_pred_y1 = reg_cpu[(j * 4 + 1) * cstep + y_index * reg_width + x_index];
                    raw_pred_w = reg_cpu[(j * 4 + 2) * cstep + y_index * reg_width + x_index];
                    raw_pred_h = reg_cpu[(j * 4 + 3) * cstep + y_index * reg_width + x_index];

                    prior_center_x = (x_index + offset) * step_w;
                    prior_center_y = (y_index + offset) * step_h;
                    center_x = variance_list[0] * raw_pred_x1 * prior_w + prior_center_x;
                    center_y = variance_list[1] * raw_pred_y1 * prior_h + prior_center_y;
                    pred_w = (std::exp(variance_list[2] * raw_pred_w) * prior_w);
                    pred_h = (std::exp(variance_list[3] * raw_pred_h) * prior_h);
                    pred_x = (center_x - pred_w / 2.);
                    pred_y = (center_y - pred_h / 2.);

                    Bbox bbox;
                    bbox.confidence = 1.0 / (1.0 + std::exp(x0 - x1));
                    MyRect pred_frect, img_frect;
                    MyRectInit(std::max(pred_x, 0.f), std::max(pred_y, 0.f), std::max(pred_w, 0.f), std::max(pred_h, 0.f), &pred_frect);
                    MyRectInit(0, 0, img_width, img_height, &img_frect);
                    MyRectIntersect(pred_frect, img_frect, &(bbox.frect));
                    if (MyRectIou(bbox.frect, pred_frect) > 0.9)
                    {
                        bbox.deleted = false;
                        vbbox.push_back(bbox);
                    }
                }
            }
        }
    }

    if (vbbox.size() != 0)
    {
        nms(vbbox, nms_thres);
    }

    std::vector<Bbox> final_vbbox;
    for (auto &bb : vbbox)
    {
        if (!bb.deleted)
        {
            final_vbbox.push_back(bb);
        }
    }

    return final_vbbox;
}

int Net::inference(object &input_object, object &output_object, int inputHeight, int inputWidth)
{

    PyArrayObject *input_data_arr = reinterpret_cast<PyArrayObject *>(input_object.ptr());
    float *input = static_cast<float *>(PyArray_DATA(input_data_arr));

    PyArrayObject *output_data_arr = reinterpret_cast<PyArrayObject *>(output_object.ptr());
    float *output = static_cast<float *>(PyArray_DATA(output_data_arr));

    ncnn::Mat in = ncnn::Mat(inputWidth, inputHeight, 3, 4u);

    memcpy(in.channel(2), input, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(1), input + inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(0), input + 2 * inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(4);
    // std::cout<<"mat in chw"<<in.c<<in.h<<in.w<<std::endl;
    // std::cout<<in.channel(0).row(0)[0]<<" "<<in.channel(1).row(0)[0]<<" "<<in.channel(2).row(0)[0]<<std::endl;
    ex.input(inputBlobNmae.c_str(), in);
    ncnn::Mat out;
    ex.extract(outputBlobName.c_str(), out);

    // for (int i=0; i<out.h; i++)
    // {
    //     const float* values = out.row(i);
    //     std::cout<<values[0]<<" "<<values[1]<<" "<<values[2]<<" "<<values[3]<<" "<<values[4]<<" "<<values[5]<<std::endl;

    // }

    for (int i = 0; i < out.c; i++)
    {
        memcpy(output + i * out.h * out.w, out.channel(i), sizeof(float) * out.h * out.w);
    }

    // printf("%d %d %d\n", out.w, out.h, out.c);
    // for (int ic=0;ic<out.c;ic++)
    // {
    //     float maxv=0;
    //     int maxx=0;
    //     int maxy=0;
    //     for(int ih=0;ih<out.h;ih++)
    //         for (int iw=0;iw<out.w;iw++)
    //         {
    //             float val=out.channel(ic).row(ih)[iw];
    //             if (maxv<val)
    //             {
    //                 maxv=val;
    //                 maxx=iw;
    //                 maxy=ih;
    //             }
    //         }
    //     printf("now value: %f(%d,%d)\n", maxv, maxx, maxy);
    // }
    // for(int h =0;h<out.h;h++)
    // {
    //     for(int w=0;w<out.w;w++)
    //     {
    //         for(int c = 0;c<out.c;c++)
    //         {
    //             *output++ = (float)out.channel(c).row(h)[w];
    //         }
    //     }
    // }

    return 0;
}

int Net::inferenceJRfaceDet(object &input_object, object &output_object, int inputHeight, int inputWidth)
{

    PyArrayObject *input_data_arr = reinterpret_cast<PyArrayObject *>(input_object.ptr());
    float *input = static_cast<float *>(PyArray_DATA(input_data_arr));

    PyArrayObject *output_data_arr = reinterpret_cast<PyArrayObject *>(output_object.ptr());
    float *output = static_cast<float *>(PyArray_DATA(output_data_arr));

    ncnn::Mat in = ncnn::Mat(inputWidth, inputHeight, 3, 4u);

    memcpy(in.channel(2), input, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(1), input + inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(0), input + 2 * inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(4);
    // std::cout<<"mat in chw"<<in.c<<in.h<<in.w<<std::endl;
    // std::cout<<in.channel(0).row(0)[0]<<" "<<in.channel(1).row(0)[0]<<" "<<in.channel(2).row(0)[0]<<std::endl;
    ex.input(inputBlobNmae.c_str(), in);
    // ncnn::Mat out;
    // ex.extract(outputBlobName.c_str(),out);

    // for(int i = 0;i<out.c;i++)
    // {
    //     memcpy(output + i*out.h*out.w,out.channel(i),sizeof(float)*out.h*out.w);
    // }
    ncnn::Mat conf_out;
    ex.extract("ConvNd_19", conf_out);
    ncnn::Mat loc_out;
    ex.extract("ConvNd_17", loc_out);
    float nms_thres = 0.35;

    std::vector<Bbox> result = process_single_output_detection(conf_out, loc_out, nms_thres);
    float *p = output;
    for (int i = 0; i < result.size(); i++)
    {
        // std::cout<<result[i].confidence<<" "<<result[i].frect.x<<" "<<result[i].frect.y<<" "<<result[i].frect.w<<" "<<result[i].frect.h<<std::endl;
        p[0] = 1.0;
        p[1] = result[i].confidence;
        p[2] = result[i].frect.x / 384;
        p[3] = result[i].frect.y / 216;
        p[4] = result[i].frect.w / 384;
        p[5] = result[i].frect.h / 216;
        p = p + 6;
    }

    return 0;
}

void extract_feature_blob_f32_debug(const char *comment, const char * imageName, const char *layer_name, const ncnn::Mat &blob);
int Net::inference_debug_writeOutputBlob2File(object &input_object,int inputHeight, int inputWidth,string typeName,int beginLayerIndex,int endLayerIndex,string imagename)
{

    PyArrayObject *input_data_arr = reinterpret_cast<PyArrayObject *>(input_object.ptr());
    float *input = static_cast<float *>(PyArray_DATA(input_data_arr));

    ncnn::Mat in = ncnn::Mat(inputWidth, inputHeight, 3, 4u);

    memcpy(in.channel(2), input, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(1), input + inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);
    memcpy(in.channel(0), input + 2 * inputWidth * inputHeight, sizeof(float) * inputWidth * inputHeight);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input(inputBlobNmae.c_str(), in);
    // ex.set_num_threads(4);
    std::cout<<"save File"<<std::endl;
    //输出各个层的结果到文件
    assert(beginLayerIndex <= endLayerIndex && beginLayerIndex >=0);
    for (int layer_index = beginLayerIndex; layer_index < endLayerIndex; layer_index++)
    {
        char layerName[128] = {'\0'};
        sprintf(layerName, "ConvNd_%d", layer_index);
        // ex.input(inputBlobNmae.c_str(), in);
        ncnn::Mat out;
        ex.extract(layerName, out);
        extract_feature_blob_f32_debug(typeName.c_str(), imagename.c_str(), layerName, out);
    }

    return 0;
}
/*
* Extract the blob feature map
*/
void extract_feature_blob_f32_debug(const char *comment, const char * imageName, const char *layer_name, const ncnn::Mat &blob)
{
    char file_path_output[128] = {'\0'};
    char file_dir[128] = {'\0'};

    FILE *pFile = NULL;

    std::string name = layer_name;

    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

    sprintf(file_dir, "./output/%s/", comment);
    mkdir(file_dir, 0777);

    sprintf(file_dir, "./output/%s/%s/", comment, imageName);
    mkdir(file_dir, 0777);

    sprintf(file_path_output, "./output/%s/%s/%s_blob_data.txt", comment, imageName, layer_name);

    pFile = fopen(file_path_output, "w");
    if (pFile == NULL)
    {
        printf("open file error!\n");
    }

    int channel_num = blob.c;

    //save top feature maps
    for (int k = 0; k < channel_num; k++)
    {
        fprintf(pFile, "blob channel %d:\n", k);

        //float *data = top_blob.data + top_blob.cstep*k;
        const float *data = blob.channel(k);
        for (int i = 0; i < blob.h; i++)
        {
            for (int j = 0; j < blob.w; j++)
            {
                fprintf(pFile, "%s%8.6f ", (data[j] < 0) ? "" : " ", data[j]);
            }
            // fprintf(pFile, "\n");
            data += blob.w;
        }
        fprintf(pFile, "\n");
    }

    //close file
    fclose(pFile);
    pFile = NULL;
}

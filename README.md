# ncnn-int8-e2e

ncnn-int8-e2e 是基于ncnn int8社区版本低比特(小于8bit)量化魔改版


# 目标

* 支持activation 7bit、weight 6bit量化
* inference time缩减为float32的50%
* x86 simulator int8 inference
* arm runtime int8 inference


---

### x86平台编译
```
./compiler.sh linux

```

### caffe2ncnn使用说明

主要区分float32与int8网络模型转换时的差异，以pose_hrnet101为例子

#### float32的ncnn模型
```
./caffe2ncnn pose_hrnet101.prototxt pose_hrnet101.caffemodel pose_hrnet101-fp32.param pose_hrnet101-fp32.bin
```

#### int8的ncnn模型
其中的pose_hrnet101.table文件由PyTorch int8 e2e量化工具生成
```
./caffe2ncnn pose_hrnet101.prototxt pose_hrnet101.caffemodel pose_hrnet101-int8.param pose_hrnet101-int8.bin 0 pose_hrnet101.table
```

#### pose example
```
./compiler.sh linux

```
编译完成后，pose位于dg-ncnn/build-linux/install/bin/
运行float32的模型
```
$ ./pose pose.jpg pose_hrnet101-fp32.param pose_hrnet101-fp32.bin 1 1 2
--- DeepGlint ncnn post demo --- 10:20:05 Apr 22 2019
iter cost 0/1: 192.42602539ms
40 40 7
now value: 0.855324(8,6)
now value: 0.746584(16,7)
now value: 0.488715(4,13)
now value: 0.600376(24,11)
now value: 0.573444(4,22)
now value: 0.556145(30,13)
now value: 0.147505(6,28)

```
运行int8的模型
```
$ ./pose pose.jpg pose_hrnet101-int8.param pose_hrnet101-int8.bin 1 1 2
--- DeepGlint ncnn post demo --- 10:20:05 Apr 22 2019
iter cost 0/1: 283.76904297ms
40 40 7
now value: 0.848378(8,6)
now value: 0.720524(16,7)
now value: 0.485292(4,13)
now value: 0.537115(24,11)
now value: 0.571759(4,22)
now value: 0.555397(30,13)
now value: 0.175053(7,28)

```



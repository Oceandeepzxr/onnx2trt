---
typora-root-url: pic
---

[TOC]



# 基于ReintaFace的TensorRT加速部署

## 概述

### RetinaFace算法

Retinaface是一种利用联合额外监督和自我监督的多任务学习方式，可以实现对不同尺度人脸进行检测的单级人脸检测器。同时，RetinaFace算法可以使用Nvidia的TensorRT推理加速技术对算法的推理速度进行优化。本次项目就是使用TensorRT的推理加速功能实现基于RetinaFace代码的快速人脸检测。

### 模型类型介绍

#### .pt/.pth

.pt/.pth文件为使用pytorch框架进行训练的模型，两种后缀都是有效的模型后缀，且除了后缀名不同之外没什么格式上的区别，一般常用.pth作为pytorch框架训练模型的后缀。

在pytorch进行模型保存时，有两种方式：

1. 保存下整个模型：使用`torch.save(model, "xxx.pth")`进行保存
2. 只保存模型的参数：使用`torch.save(model.state_dict(), "xxx.pth")`进行保存

本次RetinaFace的代码进行训练保存模型时使用的是第2种保存方式，只保存了模型的参数。因为在加载模型时需要先构建网络结构再将.pth文件中保存的参数加载进来。

#### .onnx

.onnx格式的ONNX模型时用于ML模型的开放格式，允许再不同的ML框架和工具之间交换模型。由于TensorRT不支持对pytorch框架训练的模型直接加速优化，因此需要将pytorch的模型转化为onnx模型再使用TensorRT进行加速。

ONNX模型目前也有自己的推理框架onnxruntime，方便对转化后的ONNX模型进行检验。

ONNX的github链接：https://github.com/onnx/onnx

#### .engine

.engine文件是TensorRT用于推理的文件，类似于各深度学习框架中的模型（也可使用.trt，没有内容是的区别。）

TensorRT的官方说明文档链接：https://docs.nvidia.com/deeplearning/tensorrt/index.html



## 基本部署流程

### .pth->.onnx

将pytorch的模型转化为onnx的模型可以直接使用pytorch的库中的函数torch.onnx.export()，更多关于torch.onnx的信息可以参考pytorch的官方文档：https://pytorch.org/docs/1.2.0/onnx.html#supported-operators。转换代码在RetinaFace的官方github中已经写好，需要先构建好网络结构再导入模型（模型只保存了参数）。

这里使用的版本：

- onnx==1.8.0
- onnxruntime=1.6.0

另外这个步骤需要注意的是，由于ONNX转换时的opset_version默认为9（torch.onnx.export()中的参数opset_version=），而这个opsetv9是不支持网络中的**Upsample**（上采样）层的，因此需要设置参数**opset_version=11**，即可满足这个上采样层。

生成onnx模型后，由于onnx转化模型的特点，会将网络中某些非onnx预置层的操作按最基础的步骤实现出来，导致某些非预置层特别复杂冗余，因此最好使用onnx简化工具对转化后的onnx模型进行简化。

onnx-simplifier的github链接：https://github.com/daquexian/onnx-simplifier

使用方法按照github的指导即可，简化后会有自动的ckeck环节，如果check环节出错可按照提示添加**两个--skip参数**即可，生成的模型使用netron打开发现在输入部分有一些多余的常量，不过不参与推理过程，不影响后续的转换。

onnx模型的输入尺寸为[1, 3, 640, 640]，因为训练时retinaface将图片resize成[640, 640]的尺寸，因此在转化onnx模型时将尺寸固定为该尺寸。需要注意的是由于RetinaFace训练代码中图片直接使用cv2.imread读入后喂入网络，没有对RGB通道进行处理，因此训练时的图片均为cv2默认的**BGR**通道顺序。

onnx模型的输出有三个，尺寸分别为[1, 16800, 4], [1, 16800, 10], [1, 16800, 2]分别对应要输出的边界框坐标，5个人脸特征点坐标，以及置信度。

### .onnx->.engine

将onnx模型转化为TensorRT可用的engine，本次项目使用python API，也可以使用TensorRT docker内自带的**trtexec**工具进行转换，pythonAPI的详细操作步骤见TensorRT的官方文档：https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics。

trtexec的使用方法可以参考github上TensorRT的教程：https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleOnnxMNIST。

本次项目编写的pythonAPI代码如下：

![onnx2trt_code](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\onnx2trt_code.png)

生成的tensorrt的.engine文件采用默认的**FP32**精度。

这里转换参考了github上的onnx-tensort工程，该工程github链接：https://github.com/onnx/onnx-tensorrt。代码中引用的trtutils.backend即为onnx-tensorrt工程中的backend文件。

这里需要注意的是，本项目使用的tensorRT版本为**7.1.2.8**，tensorRT7.0以后需要在builder.create_network（）中添加**EXPLICIT_BATCH**，可以参照上图所示的代码。

生成的文件为“xxx.engine”，是以及序列化之后的engine文件，可以直接用于TensorRT的推理。

### TensorRT的加速推理

使用序列化后的engine对图片进行加速推理，此处代码参考了onnx-tensorrt工程中的tensor_engine.py，并将输入和输出进行了与RetinaFace的pytorch模型推理相同的前后处理操作，保证了推理的效果最大程度不改变。

可以看到在启动TensorRT的推理engine后，推理速度非常快，平均一张图片推理时间在**3ms左右**，远远要快于pytorch推理的12ms左右。

![tensorrt_infer_spped](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\tensorrt_infer_spped.png)

推理得到结果在数值准确度方面有所下降，主要体现在置信度的略有降低，以下为pytorch、onnx以及tensorrt的模型对同一张图片进行推理的效果：

![compare](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\compare.png)



## 其他部署流程

在Retinaface的pytorch版官方github链接上给出了TensorRT部署的相关工程，github链接 ：https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface。

该方法将pytorch模型加载进来，将权重文件的参数以及网络结构写成一个.wts格式的文件，类似于文本文件，再通过手动在TensorRT中添加层来直接形成TensorRT推理加速网络进行推理。

整个部署流程可以参考github链接上的步骤，其中需要注意的一点是，使用Retinaface pytorch版本github官网上代码要使用https://github.com/wang-xinyu/Pytorch_Retinaface 中的探测代码detect.py将完整的模型保存下来

推理使用环境：

推理网络输入图片尺寸：640x640

使用tensorRT加速推理时间：

![wts2trt](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\wts2trt.png)

推理效果：



![retinaface_wts](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\retinaface_wts.jpg)

可以看到由于使用推理输入尺寸为640x640，而github链接上的同一张图片使用的输入尺寸为928x1600。因此人脸的检测框数量有一定的差距，但对于该尺寸大小的图片效果很不错，由于只推理了一张，可以看到时间在300ms左右，后续推理时间将会大幅减小。

## 问题汇总

1. TensorRT对图片进行推理时未报错但效果降低很多，如下图所示：

![problem1](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem1.png)

可能原因：Base64解码时未使用cv2工具解码，而之前测试使用cv2.imread直接读取图片，所以出现这种情况。

2. docker在安装opencv-python后仍无法使用，报错如下：

![problem2](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem2.png)

需要安装libgl1-mesa-glx

可以参考：https://blog.csdn.net/qq_35516745/article/details/103822597

3. 生成engine时的精度警告：

![problem4](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem4.png)

可以在代码中手动降低精度。

4. pycuda初始化问题

![problem3](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem3.png)

在import pycuda.driver后，还需要引用pycuda.autoinit

![prob4_sol](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\prob4_sol.png)

5. 推理engine与生成engine的设备不同，建议更换与生成engine相同的gpu进行推理

![problem5](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem5.png)

6. pycuda报错，核心放弃存储问题：

![problem7](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem7.png)

解决方法：

![problem7_sol](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\problem7_sol.png)

但该问题出现时不一定只是这个问题，可能在中间过程中报错时代码运行中断也会出现这个错误。

7. TensorRT Support Matrix：

![support_matrix](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\support_matrix.png)

相关链接：https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html



## SDK封装

本项目为了方便使用和部署，封装为SDK的形式，将代码封装为Engine类。

### 输入输出格式

#### 输入格式

为了可以正常在算法平台上部署，输入由原始的图片路径改为base64编码的图片字符串，代码中增加了对base64解码的部分（此处注意解码时使用cv2进行解码，保证图片通道为**BGR**顺序），解码部分代码如下：

![base64decode](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\base64decode.png)

#### 输出格式

代码输出为一个json文件，包含了人脸的识别框（x, y, w, h）以及特征点的识别坐标(x, y, id)

输出json：

![json_format](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\json_format.png)

### 调用方法

对engine进行初始化，向类的推理方法传入base64编码的图片字符串，得到输出，最后释放cuda使用的内存。整个流程如下图代码所示，ai算法平台上有自带的debug用sdk，以下代码可以在本地服务器上用做自己测试使用：

![run](C:\Users\use\Desktop\基于RetinaFace的TensorRT加速部署\pic\run.png)




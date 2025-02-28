[简体中文](./README.md) | [English](./README_EN.md)

## sophon-demo介绍


## 简介
Sophon Demo基于SophonSDK接口进行开发，提供一系列主流算法的移植例程。包括基于TPU-NNTC和TPU-MLIR的模型编译与量化，基于BMRuntime的推理引擎移植，以及基于BMCV/OpenCV的前后处理算法移植。

SophonSDK是算能科技基于其自主研发的深度学习处理器所定制的深度学习SDK，涵盖了神经网络推理阶段所需的模型优化、高效运行时支持等能力，为深度学习应用开发和部署提供易用、高效的全栈式解决方案。目前可兼容BM1684/BM1684X/BM1688(CV186AH)，CV186AH可以直接使用BM1688的单核模型。

## 目录结构与说明
sophon-demo提供的例子从易到难分为`tutorial`、`sample`、`application`三个模块，`tutorial`模块存放一些基础接口的使用示例，`sample`模块存放一些经典算法在SOPHONSDK上的串行示例，`application`模块存放一些典型场景的典型应用。

| tutorial                                                                  | 编程语言    | 
|---                                                                        |---         | 
| [resize](./tutorial/resize/README.md)                                     | C++/Python | 
| [crop](./tutorial/crop/README.md)                                         | C++/Python | 
| [crop_and_resize_padding](./tutorial/crop_and_resize_padding/README.md)   | C++/Python | 

| sample                                                          | 算法类别          | 编程语言    | BModel         |
|---                                                            |---               |---          | ---           |
| [LPRNet](./sample/LPRNet/README.md)                           | 车牌识别          | C++/Python | FP32/FP16/INT8 |
| [ResNet](./sample/ResNet/README.md)                           | 图像分类          | C++/Python | FP32/FP16/INT8 |
| [RetinaFace](./sample/RetinaFace/README.md)                   | 人脸检测          | C++/Python | FP32           |
| [segformer](./sample/segformer/README.md)                     | 语义分割          | C++/Python | FP32/FP16      |
| [SAM](./sample/SAM/README.md)                                 | 语义分割          | Python     | FP32/FP16      |
| [yolact](./sample/yolact/README.md)                           | 实例分割          | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_seg](./sample/YOLOv8_seg/README.md)                   | 实例分割          | C++/Python | FP32/FP16/INT8 |
| [PP-OCR](./sample/PP-OCR/README.md)                           | OCR              | C++/Python | FP32/FP16      | 
| [OpenPose](./sample/OpenPose/README.md)                       | 人体关键点检测    | C++/Python | FP32/FP16/INT8 |
| [C3D](./sample/C3D/README.md)                                 | 视频动作识别      | C++/Python | FP32/FP16/INT8 |
| [DeepSORT](./sample/DeepSORT/README.md)                       | 多目标跟踪        | C++/Python | FP32/FP16/INT8 |
| [ByteTrack](./sample/ByteTrack/README.md)                     | 多目标跟踪        | C++/Python | FP32/FP16/INT8 |
| [CenterNet](./sample/CenterNet/README.md)                     | 目标检测、姿态识别 | C++/Python | FP32/FP16/INT8 |
| [YOLOv5](./sample/YOLOv5/README.md)                           | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv34](./sample/YOLOv34/README.md)                         | 目标检测          | C++/Python | FP32/INT8      |
| [YOLOX](./sample/YOLOX/README.md)                             | 目标检测          | C++/Python | FP32/INT8      |
| [SSD](./sample/SSD/README.md)                                 | 目标检测          | C++/Python | FP32/INT8      |
| [YOLOv7](./sample/YOLOv7/README.md)                           | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_det](./sample/YOLOv8_det/README.md)                   | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv5_opt](./sample/YOLOv5_opt/README.md)                   | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [ppYOLOv3](./sample/ppYOLOv3/README.md)                       | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [ppYoloe](./sample/ppYoloe/README.md)                         | 目标检测          | C++/Python | FP32/FP16      |
| [WeNet](./sample/WeNet/README.md)                             | 语音识别          | C++/Python | FP32/FP16      | 
| [BERT](./sample/BERT/README.md)                               | 语言模型          | C++/Python | FP32/FP16      | 
| [ChatGLM2](./sample/ChatGLM2/README.md)                       | 语言模型          | C++/Python | FP16/INT8/INT4 | 
| [Llama2](./sample/Llama2/README.md)                           | 语言模型          | C++        | FP16/INT8/INT4 |
| [ChatGLM3](./sample/ChatGLM3/README.md)                       | 语言模型          | Python     | FP16/INT8/INT4 | 
| [Qwen](./sample/Qwen/README.md)                               | 语言模型          | Python     | FP16/INT8/INT4 | 
| [StableDiffusionV1.5](./sample/StableDiffusionV1_5/README.md) | 图像生成          | Python     | FP32/FP16      |

| application                                                    | 应用场景                  | 编程语言    | 
|---                                                             |---                       |---          | 
| [VLPR](./application/VLPR/README.md)                           | 多路车牌检测+识别          | C++/Python  | 
| [YOLOv5_multi](./application/YOLOv5_multi/README.md)           | 多路目标检测               | C++         | 
| [YOLOv5_multi_QT](./application/YOLOv5_multi_QT/README.md)     | 多路目标检测+QT_HDMI显示   | C++         | 

## 版本说明
| 版本    | 说明 | 
|---     |---   |
| 0.2.0  | 完善和修复文档、代码问题，新增application和tutorial模块，新增例程ChatGLM3和Qwen，SAM添加web ui，BERT、ByteTrack、C3D适配BM1688，原YOLOv8更名为YOLOv8_det并且添加cpp后处理加速方法，优化常用例程的auto_test，更新TPU-MLIR安装方式为pip |
| 0.1.10 | 修复文档、代码问题，新增ppYoloe、YOLOv8_seg、StableDiffusionV1.5、SAM，重构yolact，CenterNet、YOLOX、YOLOv8适配BM1688，YOLOv5、ResNet、PP-OCR、DeepSORT补充BM1688性能数据，WeNet提供C++交叉编译方法 |
| 0.1.9	 | 修复文档、代码问题，新增segformer、YOLOv7、Llama2例程，重构YOLOv34，YOLOv5、ResNet、PP-OCR、DeepSORT、LPRNet、RetinaFace、YOLOv34、WeNet适配BM1688，OpenPose后处理加速，chatglm2添加编译方法和int8/int4量化。|
| 0.1.8  | 完善修复文档、代码问题，新增BERT、ppYOLOv3、ChatGLM2，重构YOLOX，PP-OCR添加beam search，OpenPose添加tpu-kernel后处理加速，更新SFTP下载方式。|
| 0.1.7	 | 修复文档等问题，一些例程支持BM1684 mlir，重构PP-OCR、CenterNet例程，YOLOv5添加sail支持 |
| 0.1.6	 | 修复文档等问题，新增ByteTrack、YOLOv5_opt、WeNet例程 |
| 0.1.5	 | 修复文档等问题，新增DeepSORT例程，重构ResNet、LPRNet例程 |
| 0.1.4	 | 修复文档等问题，新增C3D、YOLOv8例程 |
| 0.1.3	 | 新增OpenPose例程，重构YOLOv5例程（包括适配arm PCIe、支持TPU-MLIR编译BM1684X模型、使用ffmpeg组件替换opencv解码等） |
| 0.1.2	 | 修复文档等问题，重构SSD相关例程，LPRNet/cpp/lprnet_bmcv使用ffmpeg组件替换opencv解码 |
| 0.1.1	 | 修复文档等问题，使用BMNN相关类重构LPRNet/cpp/lprnet_bmcv |
| 0.1.0	 | 提供LPRNet等10个例程，适配BM1684X(x86 PCIe、SoC)，BM1684(x86 PCIe、SoC) |

## 环境依赖
Sophon Demo主要依赖tpu-mlir、tpu-nntc、libsophon、sophon-ffmpeg、sophon-opencv、sophon-sail，其版本要求如下：
|sophon-demo|tpu-mlir |tpu-nntc |libsophon|sophon-ffmpeg|sophon-opencv|sophon-sail| 发布日期   |
|-------- |------------| --------|---------|---------    |----------   | ------    | --------  |
| 0.2.0  | >=1.6       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=23.10.01|
| 0.1.10 | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=23.07.01|
| 0.1.9  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=23.07.01|
| 0.1.8  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01|
| 0.1.7  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01|
| 0.1.6  | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.05.01|
| 0.1.5  | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.03.01|
| 0.1.4  | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   | >=22.12.01|
| 0.1.3  | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   |    -      |
| 0.1.2  | Not support | >=3.1.4 | >=0.4.3 | >=0.5.0     | >=0.5.0     | >=3.2.0   |    -      |
| 0.1.1  | Not support | >=3.1.3 | >=0.4.2 | >=0.4.0     | >=0.4.0     | >=3.1.0   |    -      |
| 0.1.0  | Not support | >=3.1.3 | >=0.3.0 | >=0.2.4     | >=0.2.4     | >=3.1.0   |    -      |
> **注意**：
> 1. 不同例程对版本的要求可能存在差异，具体以例程的README为准，可能需要安装其他第三方库。
> 2. BM1688/CV186AH与BM1684X/BM1684对应的sdk不是同一套，暂时还未发布到官网上，请联系技术人员获取。

## 技术资料

请通过算能官网[技术资料](https://developer.sophgo.com/site/index.html)获取相关文档、资料及视频教程。

## 社区

算能社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

算能社区网站：https://www.sophgo.com/

算能开发者论坛：https://developer.sophgo.com/forum/index.html


## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](./CONTRIBUTING_CN.md)。

## 许可证
[Apache License 2.0](./LICENSE)
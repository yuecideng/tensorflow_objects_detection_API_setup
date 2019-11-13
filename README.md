# Tensorflow 目标检测API使用指南
本指南包括：
- 环境配置
- 训练数据准备
- 训练模型
- 模型推理
## 环境配置(Windows)
###### 以下版本经测试可以运行，如果下载了最新版本，会出现兼容问题导致程序无法运行。
- Python 3.6
- Tensorflow 1.14 (GPU版本)
`pip install tensorflow-gpu==1.14`
- CUDA 10.0 [官网下载](https://developer.nvidia.com/cuda-10.0-download-archive)
- cudnn 7.5.0 [官网下载](https://developer.nvidia.com/rdp/form/cudnn-download-survey)(需要注册账号)
- Tensorflow models:
    1. 下载python依赖：`pip install -r requirements.txt`
    2. 将官网[模型](https://github.com/tensorflow/models)文件夹`models`放进该目录下
    3. 下载Protobuf:
        4.1. 在[下载地址](https://github.com/protocolbuffers/protobuf/releases)下找到最新的`*-win32.zip`版本并下载
        4.2. 在`C:\Program Files`创建文件夹`Google Protobuf`
        4.3. 将下载的内容解压到该文件夹下
        4.4. 将`C:\Program Files\Google Protobuf\bin`加入你的系统变量
        4.5. 在终端打开`tf_objects_detection_setup/models/research/`,并运行:
        `for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.`
    4. 系统变量设置：
        在`tf_objects_detection_setup/models/research/`目录下运行：
        ```
        python setup.py build
        python setup.py install
        ```
        加入系统变量
        ```
        <PATH_TO_TF>\tf_objects_detection_setup\models\research
        <PATH_TO_TF>\tf_objects_detection_setup\models\research\slim
        ```
- 安装`LabelImg`

## 训练数据准备
#### 文件结构
```
tf_objects_detection_setup
│   README.md
│   requirements.txt    
│
└──models
│   │   official
│   │   research
│   │   samples
│   │   tutorials
│   
└───scripts
│    │   csv2tfrecord.py
│    │   dataset_util.py
│    │   label.py
│    │   xml2csv.py
│
└───workspace
    │   export_inference_graph.py
    │   model_main.py
    └───training_demo
        └──annotations
        │   label_map.pbtxt
        │   *_labels.csv
        │   *.record
        └───images
        │    └───train
        │    │   *.jpg
        │    │   *.xml
        │    │   ...
        │    └───test
        │    │   *.jpg
        │    │   *.xml
        │    │   ...
        └──pre_trained_model
        │   checkpoint
        │   frozen_inference_graph.pb
        │   model.ckpt.data-00000-of-00001
        │   model.ckpt.index
        │   model.ckpt.meta
        └──trained_inference_graphs
        │    └─── *.pb
        │    │    └───saved_model          
        │    │   checkpoint
        │    │   frozen_inference_graph.pb
        │    │   model.ckpt.data-00000-of-00001
        │    │   model.ckpt.index
        │    │   model.ckpt.meta
        │    │   pipeline.config
        └──training
        │   *.config

```
#### 数据标注
1. 将训练数据和测试数据分别放在`tf_objects_detection_setup\workspace\training_demo\images\train`和`tf_objects_detection_setup\workspace\training_demo\images\test`下
2. 使用LabelImg打开相应的文件夹进行数据标注，label命名使用英文拼写
3. 将生成的`*.xml`文件分别放入对应的文件夹下
#### 创建Label Map文件
在`tf_objects_detection_setup\workspace\training_demo\annotations`目录下，创建一个`label_map.pbtxt`文件，如
```
# id从1开始
item {
    id: 1
    name: 'experience'
}

item {
    id: 2
    name: 'money'
}

item {
    id: 3
    name: 'material'
}
```
当训练类别发生变化时，修改增删对应的名字id即可
#### 创建Tensorflow Records文件
- 把`*.xml`文件转换为`*.csv`文件：
    终端进入`tf_objects_detection_setup\scripts`目录，运行
    ```
    python xml2csv.py -i ..\workspace\training_demo\images\train -o ..\workspace\training_demo\annotations\train_labels.csv
    python xml2csv.py -i ..\workspace\training_demo\images\test -o ..\workspace\training_demo\annotations\test_labels.csv
    ```
- 把`*.csv`文件转换为`*.record`文件:
    进入`annotations\`目录，修改`label.py`文件匹配label名称和id
    ```
    # 从0开始
    LABELS = {
    'experience': 0,
    'money': 1,
    'material': 2
    }
    ```
    进入`tf_objects_detection_setup\scripts`目录，运行
    ```
    python csv2tfrecord.py  --csv_input=..\workspace\training_demo\annotations\train_labels.csv --output_path=..\workspace\training_demo\annotations\train.record --img_path=..\workspace\training_demo\images\train
    python csv2tfrecord.py  --csv_input=..\workspace\training_demo\annotations\test_labels.csv --output_path=..\workspace\training_demo\annotations\test.record --img_path=..\workspace\training_demo\images\test 
    ```

## 训练模型
#### 配置训练Pipeline文件（*.config)
1. 从[模型库](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)中下载对应的预训练模型(eg. ssdlite_mobilenet_v2_coco), 解压后放入`training_demo\pre-trained-model`里面
2. 从`tf_objects_detection_setup\models\research\object_detection\samples\configs`里选择对应的`*.config`文件，并复制到`training_demo\training`下
3. 修改相应的`*.config`文件，具体可查看文件夹下的例子
#### 运行训练脚本
进入`tf_objects_detection_setup\workspace\training_demo`,运行
`python model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/*.config`
#### 查看训练效果
进入`tf_objects_detection_setup\workspace\training_demo`,运行
`tensorboard --logdir=training\`

## 模型推理
#### 导出训练好的推理图
1. 进入`training_demo\training`目录，按时间排序记录最新的`model.ckpt-*`文件
2. 在`training_demo\`目录下，运行
`python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/*.config --trained_checkpoint_prefix training/model.ckpt-* --output_directory trained-inference-graphs/output_inference_graph`
#### 运行推理脚本
进入`training_demo\`，运行`python inference.py`,可观看摄像头实时推理效果

## Reference
[Tensorflow Object Detection API tutprial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#preparing-workspace)
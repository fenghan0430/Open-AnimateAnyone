# Open-Animate Anyone 第一阶段训练方法

## 目录

## 准备环境

> 根据`fast_env.sh`

新建个虚拟环境

```bash
conda create -n animate python=3.8.18
conda activate animate
```

其他依赖

```bash
# 先装torch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 其他依赖
pip install diffusers==0.21.4
pip install transformers==4.32.0
pip install tqdm==4.66.1
pip install omegaconf==2.3.0
pip install einops==0.6.1
pip install opencv-python==4.8.0.76
pip install Pillow==9.5.0
pip install safetensors==0.3.3
pip install decord==0.6.0
pip install wandb==0.16.1
pip install accelerate==0.22.0
pip install av==11.0.0
pip install imageio==2.9.0
pip install imageio-ffmpeg
pip install gradio==3.41.2
pip install xformers==0.0.16
```

下载预训练模型和clip

```bash
cd pretrained_models
# 记得装lfs
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
git clone https://huggingface.co/openai/clip-vit-base-patch32
```

## 准备数据

### 下载数据集

下载ubc-fashion数据集，来自`https://vision.cs.ubc.ca/datasets/fashion/`

> 吐槽UBC，UBC把数据集上传到了亚马逊云。写了一个简易的下载脚本，但是下载脚本过于简易，甚至没有中断后继续下载。
> 
> 如果无法下载，推荐去我的百度云下载。

```bash
链接：https://pan.baidu.com/s/1kqxGp_8Iboz-nZbjxJTt-w 
提取码：ai00
```

```bash
# 文件结构如下
ubc-fashion-dataset
|-train/
|  |-视频
|-test/
|  |-视频
|-其他...
```

训练集和测试集一共600个视频。下载完成后把数据集移动到项目根目录下

```sh
mv ./ubc-fashion-dataset 你的项目位置
```

### 制作csv

data文件夹下有make_csv.py脚本，这个是制作csv的脚本

在`make_csv.py`把`dataset_folder`设置为视频的路径，把`csv_path`设置为生成的csv的输出路径。

**注意**运行这个脚本需要在`data/`下，因为生成的csv文件需要保存在`data/`目录

```bash
cd data/

python make_csv.py
```

脚本运行完成后，会有`UBC_train_info.csv`和`UBC_test_info.csv`两个文件。

### 制作动作序列

为ubc数据集的视频制作动作序列。该项目使用的是[DWPose](https://github.com/IDEA-Research/DWPose)。

有一些必要的模型文件要从huggingface下载[https://huggingface.co/yzd-v/DWPose/tree/main](https://huggingface.co/yzd-v/DWPose/tree/main)

将仓库克隆下来后，需要确保仓库保存到了`DWPose/`文件夹下，并改名成`dwpose_ckpts`

```bash
cd DWPose/

git clone https://huggingface.co/yzd-v/DWPose

mv DWPose dwpose_ckpts
```

这样我们就完成了模型的下载。

接着，打开`prepare_ubc.py`，修改其中的变量`dataset_folder`为你的ubc数据集的地址，随后就可以开始运行脚本。

脚本会把动作序列生成到数据集目录下的`train_dwpose`和`test_dwpose`。

> 细心的孩子会发现CPU使用率满了，没错，脚本使用CPU生成动作序列。

## 调整训练参数

### 第一阶段训练

打开第一阶段的训练配置文件`train_stage_1.yaml`，在`configs/training/`下。

从上往下调整参数：

- `pretrained_model_path`和`clip_model_path`是指v1.5模型和CLIP模型的位置，确定位置是正确的。
- `train_data`下有`csv_path`、`video_folder`和`clip_model_path`。其中`csv_path`指csv的位置；`video_folder`是指ubc数据集的位置；`clip_model_path`是CLIP的位置。

调整完这些参数就可以开始第一阶段的训练了。

## 开始训练

根据作者介绍，该项目至少需要80GB的显存才能运行。

下面是启动命令的示例：

```shell
# 单机单卡
torchrun --nproc_per_node=1 train_hack.py --config configs/training/train_stage_1.yaml

# 单机多卡
torchrun --nproc_per_node=4 train_hack.py --config configs/training/train_stage_1.yaml

# 多机多卡(正在学习)
```

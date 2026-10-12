# FeatherTalk

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg">
  <a href="https://github.com/anliyuan/FeatherTalk/stargazers"><img src="https://img.shields.io/github/stars/anliyuan/FeatherTalk?style=social"></a>
</p>

## 更新

新增 MobileOne 的训练、推理和导出入口，推理和导出自动重参数化。
音频测速现在分别报告离线摊销和 800ms 窗口重算耗时。欢迎扫码加群交流，enjoy 🎉

FeatherTalk is a lightweight personalized talking-head project. Give it a short
25 FPS video of one person, train a dedicated model, and then drive that person
with any speech audio. Its audio features come from **FeatherHuBERT**, a compact
audio encoder trained for this project through HuBERT feature distillation.

FeatherTalk 是一个轻量级个人数字人项目。准备一段单人口播视频，训练一个专属模型，
之后就可以用任意语音驱动这个人物说话。我们自行训练了轻量音频编码器
**FeatherHuBERT**，搭配个性化视觉模型，面向 CPU 和移动端部署。

## Demo / 效果

两侧使用相同的人物资源和同一段 60 秒音频。左侧是之前的旧版本，
右侧是当前更新后版本。视频对下半脸做了同比例放大，方便观察唇形和融合效果。视觉上少了很多抖动，很多音节嘴形也更准了。

<video src="./demo/feathertalk_comparison_144.mp4" controls width="100%"></video>

[打开对比视频 / Open comparison video](./demo/feathertalk_comparison_144.mp4)

## Why FeatherTalk? / 为什么选它

- **自训练音频编码器**：FeatherHuBERT 是我们通过 HuBERT 特征蒸馏训练的轻量网络，约 **3.36M** 参数。部署时无需加载大型 HuBERT 教师模型，训练好的音频权重已随仓库提供。
- **效果更好**：同时利用前后语音上下文，并针对嘴部细节和连续帧稳定性训练。
- **轻量视觉模型**：默认 FeatherTalk 视觉模型输入为 144×144，参数量约 **5.46M**。另提供可重参数化的 MobileOne 版本。
- **推理更高效**：提前提取音频特征，视频逐帧复用，不必每帧重复提取。当前提供离线推理流程，可按需求扩展流式处理。
- **真正可部署**：同时提供 Python 训练/推理和 C++/MNN 接入路线，可纯 CPU 运行。可轻松部署移动端。

## FeatherHuBERT / 自训练音频编码器

FeatherHuBERT 是独立的轻量卷积与时序卷积网络，以 HuBERT 的语音特征作为教师信号进行蒸馏训练。
它直接接收 **16kHz 音频波形**，输出 **1024 维、20ms 步长**的语音特征，供视觉模型驱动嘴部运动。
这里的 HuBERT 兼容指特征维度和时间步长，不代表两者输出数值完全相同。

仓库内置的 `feather_hubert.pth` 是已训练好的通用音频权重，可在不同人物之间复用。
训练新人物时，只需训练其视觉模型，**不需要重新训练 FeatherHuBERT**。
视觉模型内部的 Audio Adapter 负责处理这些音频特征；它与 FeatherHuBERT 是两个不同的模块。

当前公开的 FeatherHuBERT 权重仅使用中文语音训练。受作者硬盘容量限制，当前训练并不完整。
有意使用更完整的数据继续训练的开发者，可通过文末交流群联系作者，
索取 FeatherHuBERT 训练代码。

## Quick Start / 快速开始

### 1. 安装

Python 3.10 和 FFmpeg 是必需的。训练推荐使用 CUDA GPU。

```bash
git clone https://github.com/anliyuan/FeatherTalk.git
cd FeatherTalk

conda create -n feathertalk python=3.10
conda activate feathertalk
pip install -r requirements.txt
```

仓库根目录已经包含 `feather_hubert.pth` 音频编码器权重，无需额外下载。
其他权重说明见 [docs/WEIGHTS.md](docs/WEIGHTS.md)。

### 2. 准备训练视频

推荐使用 3–5 分钟、25 FPS 的单人口播视频。人脸需要完整可见，音频尽量清晰、无明显噪声和回声。

```bash
python data_utils/process.py /path/to/person/train.mp4 \
  --feather_hubert_checkpoint ./feather_hubert.pth
```

预处理会在视频所在目录生成训练需要的图像、关键点和音频特征。

### 3. 训练

```bash
python train.py \
  --dataset_dir /path/to/person \
  --save_dir /path/to/checkpoints \
  --model feathertalk \
  --epochs 200 \
  --batchsize 16
```

默认模型为 `feathertalk`。训练 MobileOne 时，把上面的参数改成 `--model mobileone`。
两者均使用 144×144 图像、FeatherHuBERT 特征和相同的预处理数据，无需重新提特征。
默认模型使用 20 帧音频上下文；MobileOne 保留原生的 8 帧窗口。

继续训练：

```bash
python train.py \
  --dataset_dir /path/to/person \
  --save_dir /path/to/checkpoints \
  --resume /path/to/checkpoints/last.pth \
  --epochs 200 \
  --batchsize 16
```

续训自动识别 checkpoint 的模型类型；`--epochs` 是总训练轮数，不是追加轮数。
请使用未重参数化的训练 checkpoint 续训，不要使用部署权重。

### 4. 提取测试音频特征

```bash
python data_utils/feather_hubert/feather_hubert.py \
  --wav /path/to/test.wav \
  --checkpoint ./feather_hubert.pth \
  --out /path/to/test_hu.npy
```

### 5. 生成视频

```bash
python inference.py \
  --dataset /path/to/person \
  --audio_feat /path/to/test_hu.npy \
  --audio_wav /path/to/test.wav \
  --checkpoint /path/to/checkpoints/199.pth \
  --save_path result.mp4
```

推理默认 `--model auto`，根据 checkpoint 自动选择模型；MobileOne 会自动重参数化。
也可显式指定 `--model feathertalk` 或 `--model mobileone`，类型不匹配时会报错。

## C++ / MNN

C++ 版本可直接读取 WAV，在 MNN 中完成音频编码、视觉模型推理和回贴合成，
运行时不依赖 Python、PyTorch、OpenCV 或 ONNX Runtime。

两种视觉模型均可导出，MobileOne 自动重参数化并适配相同的 C++ 输入协议。
测速分别报告整段音频的离线摊销耗时和 800ms 窗口逐帧重算耗时；后者仅为成本测试，
不代表已经提供完整流式接口。

模型转换、编译和工程接入说明见
[FeatherTalk-CPP/README.md](FeatherTalk-CPP/README.md)。

## Notes / 注意

- 这是个性化模型：每个人物需要单独训练一份视觉权重。
- 训练视频和收音质量会直接影响最终效果。
- 演示视频和上述 5.46M 参数量对应默认 FeatherTalk 模型，不代表 MobileOne 的效果或参数量。
- 仓库内置通用 `feather_hubert.pth`；个性化 checkpoint、ONNX 和 MNN 模型不直接提交。

## License

Apache-2.0. If FeatherTalk helps you, a star is always appreciated. 🎉

## Community / 交流群

欢迎扫码加入 UDH 数字人交流群，一起交流训练、推理、C++ 部署和二次开发。
群二维码有效期有限，过期后会在这里更新。

<p align="center">
  <img src="./assets/wechat_group_qr.jpg" alt="UDH 数字人交流群二维码" width="360">
</p>

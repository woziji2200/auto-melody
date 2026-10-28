# 训练指南

本指南面向希望检查、复现或重新训练 Auto Melody 的开发者。普通使用者只需按仓库首页的[快速开始](../README.md#快速开始)完成推理，不需要准备训练数据。

各模型的分工、节奏骨架与短语检索的设计见[项目设计](DESIGN.md)。

## 环境与数据

需要 Python 3.10+。训练建议使用配有 NVIDIA GPU 的机器；所有阶段会自动选择 CUDA 或 CPU。CPU 可以运行，但训练和 MERT 特征预处理会明显更慢。

先安装与本机匹配的 PyTorch 和 torchaudio，然后安装项目：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

将自己有权使用的 Malody `.mcz` 训练数据放入一个目录，例如 `malody-map/`。项目不提供原始歌曲或人工谱面。准备好 MERT：

```bash
python -m pip install --upgrade huggingface_hub
hf download m-a-p/MERT-v1-330M --local-dir data/MERT
```

请阅读 [MERT 模型与许可](https://huggingface.co/m-a-p/MERT-v1-330M)，并确认训练数据和数据派生产物允许按预期使用或分发。

## 预处理

这一步会扫描 MCZ、写出清单、按音频分组划分数据、解码音频缓存，并建立节奏索引和短语特征：

```bash
python preprocess.py malody-map \
  --output-dir data \
  --mert-path data/MERT \
  --device auto \
  --workers 8
```

`--device` 可设为 `auto`、`cuda` 或 `cpu`。生成的主要文件：

```text
data/manifests/all-4k.jsonl
data/splits-difficulty/{train,validation,test}.jsonl
data/audio-cache-24k/
data/rhythm-{train,validation}.pt
data/phrase-features.pt
```

这些是本地训练数据或其派生产物。若训练数据许可不允许再分发，不要把这些文件上传到公共仓库。

## 三阶段训练

按顺序运行三个阶段。第二阶段需要第一阶段产出的 checkpoint：

```bash
python train.py salience \
  --train-manifest data/splits-difficulty/train.jsonl \
  --validation-manifest data/splits-difficulty/validation.jsonl \
  --train-index data/rhythm-train.pt \
  --validation-index data/rhythm-validation.pt \
  --dataset-dir malody-map \
  --audio-cache-dir data/audio-cache-24k \
  --mert-path data/MERT \
  --output-dir runs/salience

python train.py rhythm \
  --salience-checkpoint runs/salience/best.pt \
  --mert-path data/MERT \
  --train-manifest data/splits-difficulty/train.jsonl \
  --validation-manifest data/splits-difficulty/validation.jsonl \
  --dataset-dir malody-map \
  --audio-cache-dir data/audio-cache-24k \
  --output-dir runs/rhythm

python train.py lane \
  --train-manifest data/splits-difficulty/train.jsonl \
  --validation-manifest data/splits-difficulty/validation.jsonl \
  --dataset-dir malody-map \
  --output-dir runs/lane
```

每阶段会保存 `best.pt`、配置和训练指标。各阶段参数：

```bash
python train.py salience --help
python train.py rhythm --help
python train.py lane --help
```

## 使用自训模型推理

推理需要三个阶段的 `best.pt`、MERT、训练节奏索引和短语特征。将文件放到首页列出的默认路径，或通过命令行参数指定 checkpoint 和索引位置：

```bash
python infer.py song.ogg \
  --levels 5 10 17 \
  --bpm 138 \
  --beat-zero-ms -432 \
  --salience-checkpoint runs/salience/best.pt \
  --rhythm-checkpoint runs/rhythm/best.pt \
  --lane-checkpoint runs/lane/best.pt \
  --train-index data/rhythm-train.pt \
  --feature-store data/phrase-features.pt \
  --mert-path data/MERT \
  --output generated/song.mcz
```

第二阶段 checkpoint 保存的训练机路径不要求在推理机存在；通过 `--salience-checkpoint` 显式指定第一阶段模型即可。

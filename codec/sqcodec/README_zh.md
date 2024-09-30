# Scalar Quantize Audio Codec

(简体中文|[English](./README.md))

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Scalar Quantize Audio Codec 是一个轻量级的音频编码解码器，采用标量量化算法，实现了高效的音频压缩与还原。该项目旨在为开发者提供一个简单、可扩展的音频编解码解决方案。项目代码基于 [Descript-Audio-Codec](https://github.com/descriptinc/descript-audio-codec) 修改，替换了原项目中的 VQ 部分，算法原理部分参考 [SimpleSpeech-2](https://arxiv.org/abs/2408.13893)。

## 目录

- [特性](#特性)
- [安装](#安装)
- [使用方法](#使用方法)
- [Roadmap](#roadmap)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 特性

- 基于标量量化的音频压缩算法实现
- 适用于 Diffusion / Flow Matching 等音频生成方案，缓解生成压力，提高生成效果

## 安装

你可以通过以下步骤来安装和使用该项目：

```bash
git clone https://github.com/jingzhunxue/flow_mirror.git
cd flow_mirror/codec/sqcodec
pip install -r requirements.txt
```

## 使用方法

Coming Soon...

## Roadmap

我们致力于不断改进和扩展 Scalar Quantize Audio Codec，以提供更强大和灵活的音频编码解码方案。以下是我们的开发路线图：

### 2024 年 10 月

#### 1.0 - 初始版本发布

- [x] 完成基础标量量化编解码器的实现并开源代码
- [ ] 释放 12 万小时中英文混合预训练权重
- [ ] 公开评估结果及评估代码
- [ ] 提供基础的文档和示例代码

## 贡献指南

我们欢迎任何形式的贡献！如果你有好的想法或发现了问题，请提交 [Issue](https://github.com/jingzhunxue/flow_mirror/issues) 或 [Pull Request](https://github.com/jingzhunxue/flow_mirror/pulls)。

## 许可证

该项目使用 [MIT 许可证](LICENSE)。

## 致谢

特别感谢以下项目和论文对本项目的启发和支持：

- [Descript-Audio-Codec](https://github.com/descriptinc/descript-audio-codec)
- [SimpleSpeech-2](https://arxiv.org/abs/2408.13893)

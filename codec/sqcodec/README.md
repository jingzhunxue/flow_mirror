# Scalar Quantize Audio Codec

([Simplified Chinese](./README_zh.md) | English)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Scalar Quantize Audio Codec is a lightweight audio codec that utilizes scalar quantization algorithms to achieve efficient audio compression and reconstruction. This project aims to provide developers with a simple and extensible audio codec solution. The project code is based on modifications to [Descript-Audio-Codec](https://github.com/descriptinc/descript-audio-codec), replacing the VQ section of the original project with SQ. The algorithm references the paper [SimpleSpeech-2](https://arxiv.org/abs/2408.13893).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contribution Guide](#contribution-guide)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Implements audio compression using scalar quantization algorithms
- Suitable for Diffusion / Flow Matching audio generation solutions, reducing generation overhead and improving results

## Installation

Follow these steps to install and use this project:

```bash
git clone https://github.com/jingzhunxue/flow_mirror.git
cd flow_mirror/codec/sqcodec
pip install -r requirements.txt
```

## Usage

Coming Soon...

## Roadmap

We are committed to continuously improving and expanding Scalar Quantize Audio Codec to provide more powerful and flexible audio encoding and decoding solutions. Here is our development roadmap:

### October 2024

#### 1.0 - Initial Release

- [x] Complete the basic scalar quantization codec implementation and open-source the code
- [ ] Release 120k hours of mixed pre-trained weights for Chinese and English
- [ ] Publish evaluation results and evaluation code
- [ ] Provide basic documentation and example code

## Contribution Guide

We welcome contributions of all kinds! If you have good ideas or find any issues, please submit an [Issue](https://github.com/jingzhunxue/flow_mirror/issues) or a [Pull Request](https://github.com/jingzhunxue/flow_mirror/pulls).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to the following projects and papers for their inspiration and support:

- [Descript-Audio-Codec](https://github.com/descriptinc/descript-audio-codec)
- [SimpleSpeech-2](https://arxiv.org/abs/2408.13893)

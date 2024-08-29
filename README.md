 [Update] 8.29 创建仓库， 发布 README & Roadmap

## 动机
尽管文本在互联网上是主要的语言形态，但许多场景如教学授课和医生问诊仍主要采用直接语音交流。此外，低龄儿童或不具备读写能力的人通过听说能力能够进行广泛的交流和表达，显示出纯语音交流具备足够的智能沟通能力。语音（Textless）交流天然包含丰富的表达信息，这在教育培训等场景中，相比纯粹的ASR文字转换，具有更高的信息价值。
同时，本项目也受到 OpenAI 发布的 GPT-4o 和其展示的教育场景中的演示视频展现的能力的启发。

## 团队
浙江精准学是由阿里巴巴投资，专注于提供教育相关软硬件产品（AI 辅学机）的公司。精准学 AI 团队致力于通过 AI 技术实现接近甚至超越人类教育体验的主动式学习，并力求降低技术成本，使之人人可负担。

## 背景
直接的语音端到端模型最早据我们所知，源自 Meta 的 Speechbot 系列和 GLSM。此外以下相关工作文献为我们的研究提供了宝贵的参考和实验经验：
- Spectron: Nachmani et al. (2024) 讨论了使用频谱图强化的 LLM 进行口语问题回答和语音续述的方法。[详细信息][1]
- SpiritLM: Nguyen et al. (2024) 探索了口语和书面语言模型的交错。[详细信息][2]
- GLSM: Lakhotia et al. (2021) 从原始音频中生成口语语言模型。[详细信息][3]
- AudioLM: Borsos et al. (2023) 提出了一种音频生成的语言建模方法。[详细信息][4]
- SpeechGPT: Zhang et al. (2023) 强化了大型语言模型的内在跨模态对话能力。[详细信息][5]
- SpeechFlow: Liu et al. (2024) 介绍了一种配合流匹配的语音生成预训练方法。[详细信息][6]

[1]: ~[https://arxiv.org/abs/2305.15255](https://arxiv.org/abs/2305.15255)~ "Spoken Question Answering and Speech Continuation Using Spectrogram-Powered LLM"
[2]: ~[https://arxiv.org/abs/2402.05755](https://arxiv.org/abs/2402.05755)~ "SpiRit-LM: Interleaved Spoken and Written Language Model"
[3]: ~[https://arxiv.org/abs/2102.01192](https://arxiv.org/abs/2102.01192)~ "Generative Spoken Language Modeling from Raw Audio"
[4]: ~[https://arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143)~ "AudioLM: a Language Modeling Approach to Audio Generation"
[5]: ~[https://arxiv.org/abs/2305.11000](https://arxiv.org/abs/2305.11000)~ "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities"
[6]: ~[https://arxiv.org/abs/2310.16338](https://arxiv.org/abs/2310.16338)~ "Generative Pre-training for Speech with Flow Matching"


## 方法
针对中文特别是支持教育场景语汇的自监督预训练语音编码器的缺乏，我们基于 HuBERT 论文的方法开发了一个侧重语义信息的 SSL 语音编码器，并借鉴 RVQVAE 的方法，使用大量中文语音数据从头训练了侧重声学信息的音频编解码器（9层码本）。
基于这些自监督预训练的编解码器，我们使用 qwen2 系列 LLM 模型作为初始化参数，在 FlowMirror-s v0.1 和 v0.2 平台上使用了 2 万小时和 5 万小时的语音数据进行端到端的预训练，并支持 ASR、TTS、语音续写、语音对话等任务。这些实验结果初步验证了语音端到端模型的可行性，并且显示出网络设计的可扩展性，预示着模型在后续版本中能够获得更强的能力。
【TODO: 插入模型结构图】

## 评估
定性音频的例子可以参考如下对话
【插入音频 mp3】
相应的 Demo 体验部署在 https://flow.jzx.ai ，限于资源有限，同时支持并发小于10。实际部署的 checkpoint 是 心流知镜-s v0.2-240822-checkpoint，后续会更新到 v0.2和 v0.3的最新的版本。

经过多任务训练的模型的ASR任务能力，大约相当于 Whisper-small 的水平。
AudioBench 的评估数据待添加
构建中文版的 AudioBench 评估

## 限制与缺点
* 在 3 个数据阶段的训练中，我们没有使用常规的文本 LLM 预训练数据，预见到与原始 qwen2 模型相比，在 MMLU 评估上可能会有能力下降。未来版本将尝试减少这种下降。
* 当前版本仅对说话人音色进行了控制，其他语音信息如情感、韵律、语速、停顿、非言语声音、音高等未进行针对性调优。
* 对话有时会答非所问，或者回答错误的话题（例如语音特有的同音字造成的误解）。当前阶段限于1.5B 的参数量，以及预训练语音数据的特殊分布（不是均匀分布在各种对话话题）以及数据预处理的瓶颈的原因，我们认为随着数据量的增加以及针对性数据的加入，会大幅度改善这一问题。
* 当前版本还不支持多轮对话。
* 推理速度还有非常大的探索空间。预计在针对 TensorRT 适配，以及一些其他流行技术的应用后，即使不考虑量化仍然有十几倍的加速空间存在。

⠀
## 许可证
由于在 v0.1 - v0.3 的自监督 Encoder 中使用了 WenetSpeech 的数据集，我们发布的自监督预训练语音 encoder 和端到端 checkpoint 仅限于学术使用。代码部分则遵循 Apache 2.0协议。
为了促进中文及亚洲地区语言的语音模型探索，我们将整理采集的公域数据，排除 Wenet 数据后训练一个新的版本，开放可以更加自由使用的自监督编码器和编解码器。

## Roadmap
### 2024-8
**心流知镜-s v0.1 & 0.2 (5亿-15亿参数)**
* 中文版自监督 audio codec
* 心流知镜-s v0.1 & v0.2 (5亿-15亿 参数)
* 基于 webrtc 的体验网站
* 语音 & 文字 双输出

⠀
### 2024-9
**心流知镜-s v0.2**
* 开源 checkpoint 和推理代码
* 推理加速版本
* 支持端侧部署
* 开放自监督 speech encoder 和 Audio codec 权重和代码供学术使用

⠀
### 2024-10
**心流知镜-s v0.3**
* 中小学科目教学增强
* 支持对话Speaker语音选择
* 语音 Expressive 表达（情绪、音量、高音、语速等）

⠀
### 2024-11
**心流知镜-s v0.3-多语言版本**
* 支持东亚地区及全球主流语言
* 支持多语种交互对话

⠀
### 2024-12
**心流知镜-s v0.4**
* 支持高品质的教育教学场景全双工对话
* 更大参数量的模型尺寸

⠀
### 2025-1
**心流知镜-s v0.5**
* 对于中国各地方言及口音的支持

⠀
# 2025-3
**心流知镜-s1**
* 发布更大参数量的模型尺寸
* 对于视觉能力的扩展

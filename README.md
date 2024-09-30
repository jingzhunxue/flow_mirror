([简体中文](./README_zh.md)|English)

[![huggingface](https://img.shields.io/badge/huggingface-ckpt-yellow)](https://huggingface.co/jzx-ai-lab/flow_mirror)
[![modelscope](https://img.shields.io/badge/modelscope-ckpt-purple)](https://www.modelscope.cn/models/jzx-ai-lab/Flow_mirror)
[![github](https://img.shields.io/badge/Github-code-black)](https://github.com/jingzhunxue/flow_mirror)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Update]  
8.29: Created repository, published README & Roadmap    
8.31: Released Demo Site (https://voice-playground.91jzx.cn)    
9.02: Released Inference Code   
9.12: Released FlowMirror-s-v0.2-checkpoint-20240828   
9.20: On-Device Inference supported by Sophgo.ai, Thanks!   
9.30: Released Speech Tokenizer SQ Codec in Dual Language Chinese/English   


## Motivation
While text remains the dominant form of language on the internet, many scenarios, such as teaching and medical consultations, still rely on direct verbal communication. Moreover, young children and individuals without literacy skills can engage in extensive communication and expression through listening and speaking, demonstrating that pure voice-based communication can provide sufficient intelligence for interaction. Spoken (textless) communication inherently contains rich expressive information, making it more valuable than purely ASR-converted text in scenarios like education and training.

Additionally, this project draws inspiration from the capabilities demonstrated by OpenAI's GPT-4 and its educational use cases showcased in demo videos.


## Team
Zhejiang Jingzhunxue is a company funded by Alibaba, focusing on providing education-related hardware and software products (AI-assisted learning devices). The AI team at Jingzhunxue is dedicated to achieving proactive learning experiences comparable to or surpassing human education using AI technologies, while striving to reduce technical costs to make these solutions affordable for everyone.


## Background
To the best of our knowledge, the earliest end-to-end voice models originated from Meta’s Speechbot GLSM series. Several relevant research papers have provided valuable references and experimental experiences for our work:
- SpiritLM: Nguyen et al. (2024) explored the interleaving of spoken and written language models.[More Info][1]
- GLSM: Lakhotia et al. (2021) Lakhotia et al. (2021) developed a generative spoken language model from raw audio.[More Info][2]
- AudioLM: Borsos et al. (2023) proposed a language modeling approach to audio generation.[More Info][3]
- SpeechGPT: Zhang et al. (2023) enhanced the cross-modal conversational capabilities of large language models.[More Info][4]
- SpeechFlow:Liu et al. (2024) introduced a speech generation pretraining method using flow matching. [More Info][5]

[1]: https://arxiv.org/abs/2402.05755 "SpiRit-LM: Interleaved Spoken and Written Language Model"
[2]: https://arxiv.org/abs/2102.01192 "Generative Spoken Language Modeling from Raw Audio"
[3]: https://arxiv.org/abs/2209.03143 "AudioLM: a Language Modeling Approach to Audio Generation"
[4]: https://arxiv.org/abs/2305.11000 "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities"
[5]: https://arxiv.org/abs/2310.16338 "Generative Pre-training for Speech with Flow Matching"


## Methodology
Overall, we view the pre-training of end-to-end voice models as a process of learning representations that capture both semantic and acoustic information inherent in speech. Initializing with a text-based LLM brings the possibility of learning unified Text & Audio Representations and significantly reduces engineering complexity. Thus, we designed the overall training process in two stages as outlined below.  

Due to the lack of self-supervised pre-trained speech encoders supporting Chinese, particularly for educational vocabulary, we developed a self-supervised speech encoder focusing on semantic information, based on the Meta HuBERT paper. Drawing inspiration from RVQVAE, we trained an audio codec focusing on acoustic information (9 layers of codebooks) from scratch using extensive Chinese speech data.
![Self-supervised Audio Codec Modeling](assets/flow_mirror_s_v02_ssl_codec.png)

Based on these self-supervised pre-trained codecs, we used the qwen2 series LLM models as initialization parameters. As shown in the figure, we adopted an asymmetric structure, where input is primarily a Semantic Unit, and output includes both Acoustic Units and text.
![Overall Architecture](assets/flow_mirror_s_v02_architecture.png)

FlowMirror-s v0.1 and v0.2 were pre-trained with 20,000 hours and 50,000 hours of speech data, respectively, and support tasks such as ASR, TTS, speech continuation, and voice dialogue. These experimental results preliminarily verify the feasibility of end-to-end voice models and demonstrate the scalability of the network design, suggesting that the model will achieve even stronger capabilities in future versions.

## Evaluation
Qualitative audio examples can be referenced through the following dialogues:
```text
example_1 = "人在没有目标的时候才应该有压力"
example_2 = "这个阶段需要学习什么知识？"
example_3 = "怎么把事情做对要花时间去培养"
example_4 = "这里的药材长势不错"
```

### Dialogue Voice Examples
**Example 1:** "People should only feel pressure when they lack a goal."  
[Input](assets/question_example_1_MP3.mp3)  
[Output](assets/answer_example_1_MP3.mp3)

**Example 2:** "The growth of the herbs here looks promising."  
[Input](assets/question_example_4_MP3.mp3)  
[Output](assets/answer_example_4_MP3.mp3)

### Demo Site
The demo is deployed at https://voice-playground.91jzx.cn, with support for up to 10 concurrent users due to limited resources. The checkpoint currently deployed is 心流知镜-s v0.2-240822-checkpoint. Future versions will update to the latest v0.2 and v0.3 checkpoints.

### Multi-task Evaluation
In this project, the ASR sub-task is considered an evaluation of how well learnable semantic information in the speech is captured during pre-training. The current checkpoint achieves ASR performance approximately equivalent to Whisper-small during the first stage of pre-training. The evaluation data consists of publicly available online speech data, which was not used during training, and Wenet data, which did not participate in end-to-end training. A random sample of 1,024 sentences from both datasets was evaluated.  
| Dataset Source            | Quantity  | Chinese CER/WER   |
|--------------------------|-----------|-------------------|
| Public Dataset - Test     | 1,024     | 12.55%            |
| WenetSpeech - Test        | 1,024     | 24.23%            |

Since this checkpoint is from an early epoch, it is expected that with increased training data and time, the alignment between speech semantics and text will significantly improve, even without increasing the model size.

**[TODO]**  
Evaluation data from AudioBench will be added.  
Note: There is an urgent need to construct a Chinese version of AudioBench for more comprehensive evaluations.

## Limitations and Drawbacks
* During the three-stage training process, we did not use conventional text LLM pre-training data. Compared to the original qwen2 model, this may lead to decreased performance in MMLU evaluations. Future versions will aim to mitigate this.
* The current version only controls the speaker's voice timbre. Other speech characteristics such as emotion, prosody, speaking rate, pauses, non-verbal sounds, and pitch have not been fine-tuned.
* Sometimes, the dialogue responses may be irrelevant or address the wrong topic (e.g., misinterpretations caused by homophones in speech). At this stage, due to the limited parameter size (1.5B) and the special distribution of pre-training speech data (not evenly distributed across conversation topics), as well as bottlenecks in data preprocessing, we anticipate significant improvements in this area with increased and more targeted data.
* Multi-turn conversations are not yet supported in the current version.
* There is substantial room for improving inference speed. The current TTFB on an L20 GPU is around 670ms. We expect that with TensorRT optimization and the application of other popular techniques, overall throughput can be improved by an order of magnitude, even without quantization.

## License
Since WenetSpeech data was used in the self-supervised encoder for v0.1-v0.3, the self-supervised pre-trained speech encoder and end-to-end checkpoint weight files are limited to academic use. The code is licensed under Apache 2.0.  
To further promote the exploration of speech models for Chinese and Asian languages, we plan to release a new version trained on publicly collected data (excluding Wenet), providing a self-supervised encoder and decoder that is more freely usable.

## Roadmap
The project is planned as follows:

### August 2024
**心流知镜-s v0.1 & 0.2 (500M-1.5B parameters)**  
- [x] Chinese self-supervised audio codec  
- [x] 心流知镜-s v0.1 & v0.2 (500M-1.5B parameters)  
- [x] Experience website based on WebRTC  
- [x] Dual output: Speech & Text  

⠀
### September 2024
**心流知镜-s v0.2**  
- [x] Open-source [checkpoint](https://huggingface.co/jzx-ai-lab/flow_mirror) and inference code  
- [ ] Accelerated inference version  
- [x] Support for on-device deployment  
- [x] Release self-supervised audio codec weights for academic use  

⠀
### October 2024
**心流知镜-s v0.3**  
- [ ] Enhanced for primary and secondary school subject teaching  
- [ ] Support for speaker voice selection in dialogues  
- [ ] Expressive speech (emotion, volume, pitch, speech rate, etc.)  
- [ ] Construction of a Chinese-focused AudioBench evaluation dataset  

⠀
### November 2024
**心流知镜-s v0.3 - Multilingual Version**  
- [ ] Support for major languages in East Asia and globally  
- [ ] Support for multilingual interactive dialogues  

⠀
### December 2024
**心流知镜-s v0.4**  
- [ ] Support for high-quality, fully duplex dialogues in educational scenarios  
- [ ] Larger model sizes  

⠀
### January 2025
**心流知镜-s v0.5**  
- [ ] Support for various Chinese dialects and accents  

⠀
### March 2025
**心流知镜-s1**  
- [ ] Release of larger model sizes  
- [ ] Expansion to visual capabilities  

## Recruitment
We are hiring for the following areas, including group leader roles. Interested candidates are welcome to apply:
- Speech ASR/TTS/Dialog SLLM  
- Role-playing LLM model  
- Multimodal model inference acceleration  
- Visual understanding and document intelligence  
- General framework for character video generation  

## Community
DingTalk Group: 90720015617  
<img src="assets/dingding_qrcode.png" alt="DingTalk Technical Group QR Code" width="200"/>

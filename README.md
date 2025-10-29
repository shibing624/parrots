[**🇨🇳中文**](https://github.com/shibing624/parrots/blob/master/README.md) | [**🌐English**](https://github.com/shibing624/parrots/blob/master/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/parrots/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
    <a href="https://github.com/shibing624/parrots">
    <img src="https://github.com/shibing624/parrots/blob/master/docs/parrots_icon.png" alt="Logo" height="156">
    </a>
    <br/>
    <br/>
    <a href="https://huggingface.co/spaces/shibing624/parrots" target="_blank"> Online Demo </a>
    <br/>
    <img width="100%" src="https://github.com/shibing624/parrots/blob/master/docs/hf.jpg">
</div>


-----------------

# Parrots: ASR and TTS toolkit
[![PyPI version](https://badge.fury.io/py/parrots.svg)](https://badge.fury.io/py/parrots)
[![Downloads](https://static.pepy.tech/badge/parrots)](https://pepy.tech/project/parrots)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/parrots.svg)](https://github.com/shibing624/parrots/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.7%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/parrots.svg)](https://github.com/shibing624/parrots/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)

## Introduction
Parrots, Automatic Speech Recognition(**ASR**), Text-To-Speech(**TTS**) toolkit, support Chinese, English, Japanese, etc.

**parrots**实现了语音识别和语音合成模型一键调用，开箱即用，支持中英文。

## Features
1. **ASR**：基于`distilwhisper`实现的中文语音识别（ASR）模型，支持中、英等多种语言
2. **TTS**：基于`GPT-SoVITS`训练的语音合成（TTS）模型，支持中、英、日等多种语言
3. **流式TTS**：支持流式语音合成，实现低延迟的实时语音输出




## Install
```shell
pip install torch # or conda install pytorch
pip install -r requirements.txt
pip install parrots
```
or
```shell
pip install torch # or conda install pytorch
git clone https://github.com/shibing624/parrots.git
cd parrots
python setup.py install
```

## Demo
- Offical Demo: https://www.mulanai.com/product/tts/
- HuggingFace Demo: https://huggingface.co/spaces/shibing624/parrots

<img width="85%" src="https://github.com/shibing624/parrots/blob/master/docs/hf.png">

run example: [examples/tts_gradio_demo.py](https://github.com/shibing624/parrots/blob/master/examples/tts_gradio_demo.py) to see the demo:
```shell
python examples/tts_gradio_demo.py
```

## Usage
### ASR(Speech Recognition)
example: [examples/demo_asr.py](https://github.com/shibing624/parrots/blob/master/examples/demo_asr.py)
```python
import os
import sys

sys.path.append('..')
from parrots import SpeechRecognition

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    m = SpeechRecognition()
    r = m.recognize_speech_from_file(os.path.join(pwd_path, 'tushuguan.wav'))
    print('[提示] 语音识别结果：', r)

```

output:
```
{'text': '北京图书馆'}
```

### TTS(Speech Synthesis)

#### 基础用法
example: [examples/demo_tts.py](https://github.com/shibing624/parrots/blob/master/examples/demo_tts.py)
```python
from parrots import TextToSpeech

# 初始化 TTS 模型（无需手动配置路径）
m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
    device="cpu",  # 或 "cuda" 使用 GPU
    half=False     # 设置为 True 使用半精度加速
)

# 生成语音
m.predict(
    text="你好，欢迎来到北京。这是一个合成录音文件的演示。Welcome to Beijing!",
    text_language="auto",  # 自动检测语言，也可指定 "zh", "en", "ja"
    output_path="output_audio.wav"
)
```

output:
```
Save audio to output_audio.wav
```

#### 流式 TTS（低延迟）

支持流式语音合成，适用于实时对话场景：

```python
from parrots import TextToSpeech
import soundfile as sf
import numpy as np

m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
)

# 流式生成语音
audio_chunks = []
for audio_chunk in m.predict_stream(
    text="这是一段较长的文本，将会被流式合成为语音。",
    text_language="zh",
    stream_chunk_size=20  # 控制延迟，越小延迟越低
):
    audio_chunks.append(audio_chunk)
    # 这里可以实时播放 audio_chunk

# 保存完整音频
full_audio = np.concatenate(audio_chunks)
sf.write("streaming_output.wav", full_audio, m.sampling_rate)
```

#### 日志管理

控制日志输出级别：

```python
from parrots import TextToSpeech
from parrots.log import set_log_level, logger

# 设置日志级别
set_log_level("INFO")  # 可选: DEBUG, INFO, WARNING, ERROR

m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
)

# 使用 logger
logger.info("开始语音合成...")
m.predict(
    text="你好，世界！",
    text_language="zh",
    output_path="output.wav"
)
```


### 命令行模式（CLI）

支持通过命令行方式执行ARS和TTS任务，代码：[cli.py](https://github.com/shibing624/parrots/blob/master/parrots/cli.py)

```
> parrots -h                                    

NAME
    parrots

SYNOPSIS
    parrots COMMAND

COMMANDS
    COMMAND is one of the following:

     asr
       Entry point of asr, recognize speech from file

     tts
       Entry point of tts, generate speech audio from text

```

run：

```shell
pip install parrots -U
# asr example
parrots asr -h
parrots asr examples/tushuguan.wav

# tts example
parrots tts -h
parrots tts "你好，欢迎来北京。welcome to the city." output_audio.wav
```

- `asr`、`tts`是二级命令，asr是语音识别，tts是语音合成，默认使用的模型是中文模型
- 各二级命令使用方法见`parrots asr -h`
- 上面示例中`examples/tushuguan.wav`是`asr`方法的`audio_file_path`参数，输入的音频文件（required）

## Release Models

### ASR
- [BELLE-2/Belle-distilwhisper-large-v2-zh](https://huggingface.co/BELLE-2/Belle-distilwhisper-large-v2-zh)
### TTS

- [shibing624/parrots-gpt-sovits-speaker](https://huggingface.co/shibing624/parrots-gpt-sovits-speaker)

| speaker name | 说话人名 | character | 角色特点 | language | 语言 |
|--|--|--|--|--|--|
| KuileBlanc | 葵·勒布朗 | lady | 标准美式女声 | en | 英 |
| LongShouRen | 龙守仁 | gentleman | 标准美式男声 | en | 英 |
| MaiMai | 卖卖| singing female anchor | 唱歌女主播声 | zh | 中 |
| XingTong | 星瞳 | singing ai girl | 活泼女声 | zh | 中 |
| XuanShen | 炫神 | game male anchor | 游戏男主播声 | zh | 中 |
| KusanagiNene | 草薙寧々 | loli | 萝莉女学生声 | ja | 日 |

- [shibing624/parrots-gpt-sovits-speaker-maimai](https://huggingface.co/shibing624/parrots-gpt-sovits-speaker-maimai)

| speaker name | 说话人名 | character | 角色特点 | language | 语言 |
|--|--|--|--|--|--|
| MaiMai | 卖卖| singing female anchor | 唱歌女主播声 | zh | 中 |

## 更新日志

### v0.2.0 (2025-10)
- ✨ 新增流式 TTS 功能，支持低延迟实时语音合成
- ✨ 新增统一的日志管理系统（基于 loguru）
- 🐛 修复 PyTorch 2.0+ 的 `weight_norm` 弃用警告
- 🐛 修复 `torch.stft` 的 `return_complex=False` 弃用警告
- 🐛 修复 librosa 的 `resample` 和 `time_stretch` 警告
- 🔧 优化模型加载机制，无需手动添加 `sys.path`
- 📝 完善文档和示例代码

### v0.1.0 (2024-12)
- 🎉 初始版本发布
- ✨ 支持 ASR（语音识别）
- ✨ 支持 TTS（语音合成）
- ✨ 支持中、英、日多语言

## Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/parrots.svg)](https://github.com/shibing624/parrots/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624*, 进Python-NLP交流群，备注：*姓名-公司名-NLP*

<img src="docs/wechat.jpeg" width="200" />


## Citation

如果你在研究中使用了parrots，请按如下格式引用：

```latex
@misc{parrots,
  title={parrots: ASR and TTS Tool},
  author={Ming Xu},
  year={2024},
  howpublished={\url{https://github.com/shibing624/parrots}},
}
```

## License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加parrots的链接和授权协议。


## Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


## Reference
#### ASR(Speech Recognition)
- [EAT: Enhanced ASR-TTS for Self-supervised Speech Recognition](https://arxiv.org/abs/2104.07474)
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
- [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
#### TTS(Speech Synthesis)
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- [keonlee9420/Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2)
- [TensorSpeech/TensorflowTTS](https://github.com/TensorSpeech/TensorflowTTS)
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

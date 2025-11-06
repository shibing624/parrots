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
1. **ASR**: Automatic Speech Recognition based on `distilwhisper`, supporting Chinese, English, and other languages
2. **TTS**: Text-to-Speech based on `GPT-SoVITS`, supporting Chinese, English, Japanese, and other languages
3. **IndexTTS2**: Integrated IndexTTS2 model for emotionally expressive and duration-controlled zero-shot speech synthesis
   - Precise speech duration control
   - Emotion and speaker identity disentanglement for independent control
   - Multiple emotion control methods: audio reference, emotion vectors, text descriptions
   - Highly expressive emotional speech synthesis
4. **Streaming TTS**: Support for streaming speech synthesis with low latency for real-time audio output




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

#### GPT-SoVITS Basic Usage
example: [examples/demo_tts.py](https://github.com/shibing624/parrots/blob/master/examples/demo_tts.py)
```python
from parrots import TextToSpeech

# Initialize TTS model (no manual path configuration needed)
m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
    device="cpu",  # or "cuda" for GPU
    half=False     # Set to True for half precision acceleration
)

# Generate speech
m.predict(
    text="Hello, welcome to Beijing. This is a demo of synthesized audio. Welcome to Beijing!",
    text_language="auto",  # Auto-detect language, or specify "zh", "en", "ja"
    output_path="output_audio.wav"
)
```

output:
```
Save audio to output_audio.wav
```

#### Streaming TTS (Low Latency)

Support for streaming speech synthesis, suitable for real-time conversation scenarios:

```python
from parrots import TextToSpeech
import soundfile as sf
import numpy as np

m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
)

# Stream generate speech
audio_chunks = []
for audio_chunk in m.predict_stream(
    text="This is a longer text that will be synthesized into speech in a streaming manner.",
    text_language="en",
    stream_chunk_size=20  # Control latency, smaller = lower latency
):
    audio_chunks.append(audio_chunk)
    # You can play audio_chunk in real-time here

# Save complete audio
full_audio = np.concatenate(audio_chunks)
sf.write("streaming_output.wav", full_audio, m.sampling_rate)
```

#### Log Management

Control log output level:

```python
from parrots import TextToSpeech
from parrots.log import set_log_level, logger

# Set log level
set_log_level("INFO")  # Options: DEBUG, INFO, WARNING, ERROR

m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
)

# Use logger
logger.info("Starting speech synthesis...")
m.predict(
    text="Hello, world!",
    text_language="en",
    output_path="output.wav"
)
```

#### IndexTTS2 Advanced Usage

IndexTTS2 is a breakthrough model for emotionally expressive and duration-controlled autoregressive zero-shot speech synthesis.

example: [examples/demo_indextts.py](https://github.com/shibing624/parrots/blob/master/examples/demo_indextts.py)

**1. Basic Voice Cloning (Single Reference Audio)**

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "Hello, welcome to Beijing. This is a demo of synthesized audio."
tts.infer(text=text, output_path="gen.wav", verbose=True)
```

**2. Emotional Speech Synthesis (With Emotion Reference Audio)**

Use a separate emotional reference audio to control the emotional expression:

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "The tavern is unconscionable, starting to auction rooms, ah, a bunch of fools."
tts.infer(
   speak_reference_audio_path_or_name='examples/voice_07.wav',  # Speaker timbre reference
   text=text,
   output_path="gen.wav",
   emo_reference_audio_path="examples/emo_sad.wav",  # Emotion reference audio
   verbose=True
)
```

**3. Adjust Emotion Intensity**

Control emotion influence with `emo_alpha` parameter (range 0.0-1.0):

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "The tavern is unconscionable, starting to auction rooms, ah, a bunch of fools."
tts.infer(
   speak_reference_audio_path_or_name='examples/voice_07.wav',
   text=text,
   output_path="gen.wav",
   emo_reference_audio_path="examples/emo_sad.wav",
   emo_alpha=0.6,  # 60% emotion intensity
   verbose=True
)
```

**4. Emotion Vector Control**

Directly provide an 8-dimensional emotion vector for precise control, in order:
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "Wow! This drop rate is so high! I'm blessed by luck!"
tts.infer(
   speak_reference_audio_path_or_name='examples/voice_10.wav',
   text=text,
   output_path="gen.wav",
   emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0],  # Surprised emotion
   use_random=False,
   verbose=True
)
```

**5. Text-Based Emotion Control**

Enable `use_emo_text` to automatically infer emotions from text content:

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "Hide quickly! He's coming! He's coming to catch us!"
tts.infer(
   speak_reference_audio_path_or_name='examples/voice_12.wav',
   text=text,
   output_path="gen.wav",
   emo_alpha=0.6,
   use_emo_text=True,  # Enable text emotion analysis
   use_random=False,
   verbose=True
)
```

**6. Independent Emotion Text Description**

Provide a separate emotion description via `emo_text` parameter:

```python
from parrots.indextts import IndexTTS2

tts = IndexTTS2()
text = "Hide quickly! He's coming! He's coming to catch us!"
emo_text = "You scared me to death! Are you a ghost?"  # Independent emotion description
tts.infer(
   speak_reference_audio_path_or_name='examples/voice_12.wav',
   text=text,
   output_path="gen.wav",
   emo_alpha=0.6,
   use_emo_text=True,
   emo_text=emo_text,
   use_random=False,
   verbose=True
)
```

**Pinyin Control Notes:**

IndexTTS2 supports mixed modeling of Chinese characters and Pinyin. When precise pronunciation control is needed, provide text with specific Pinyin annotations.
Note: Pinyin control only supports valid Chinese Pinyin combinations.

Example:
```python
text = "之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。"
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

### IndexTTS2
- [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) - Latest emotion expression and duration control model
- [IndexTeam/IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) - Improved stability and English performance
- [IndexTeam/Index-TTS](https://huggingface.co/IndexTeam/Index-TTS) - Initial version

Related Papers:
- [IndexTTS2 Paper](https://arxiv.org/abs/2506.21619) - Breakthrough in emotion expression and duration control
- [IndexTTS Paper](https://arxiv.org/abs/2502.05512) - Industrial-level controllable zero-shot TTS

### GPT-SoVITS TTS

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
- [IndexTeam/IndexTTS](https://github.com/index-tts/index-tts) - IndexTTS2 emotion expression and duration control
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- [keonlee9420/Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2)
- [TensorSpeech/TensorflowTTS](https://github.com/TensorSpeech/TensorflowTTS)
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## Changelog

### v0.3.0 (2025-11)
- 🔥 Integrated IndexTTS2 model for emotionally expressive and duration-controlled zero-shot speech synthesis
- ✨ Support multiple emotion control methods: audio reference, emotion vectors, text descriptions
- ✨ Implemented emotion and speaker identity disentanglement for independent control
- ✨ Support Pinyin mixed modeling for precise pronunciation control
- 🐛 Fixed transformers 4.50+ compatibility issues
- 🐛 Fixed dictionary parameter access errors
- 📝 Added IndexTTS2 usage examples and documentation

### v0.2.0 (2025-10)
- ✨ Added streaming TTS feature with low-latency real-time speech synthesis
- ✨ Added unified logging system (based on loguru)
- 🐛 Fixed PyTorch 2.0+ `weight_norm` deprecation warning
- 🐛 Fixed `torch.stft` `return_complex=False` deprecation warning
- 🐛 Fixed librosa `resample` and `time_stretch` warnings
- 🔧 Optimized model loading mechanism, no need to manually add `sys.path`
- 📝 Improved documentation and example code

### v0.1.0 (2024-12)
- 🎉 Initial release
- ✨ Support for ASR (Automatic Speech Recognition)
- ✨ Support for TTS (Text-to-Speech)
- ✨ Support for Chinese, English, and Japanese

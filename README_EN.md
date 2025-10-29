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
3. **Streaming TTS**: Support for streaming speech synthesis with low latency for real-time audio output




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

#### Basic Usage
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

| MaiMai | 卖卖| singing female anchor | 唱歌女主播声 | zh | 中 |

## Changelog

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

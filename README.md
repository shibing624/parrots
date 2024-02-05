[**🇨🇳中文**](https://github.com/shibing624/parrots/blob/master/README.md) | [**🌐English**](https://github.com/shibing624/parrots/blob/master/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/parrots/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/parrots">
    <img src="https://github.com/shibing624/parrots/blob/master/docs/parrots_icon.png" alt="Logo" height="156">
  </a>
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
1. ASR：基于`distilwhisper`实现的中文语音识别（ASR）模型，支持中、英等多种语言
2. TTS：基于`GPT-SoVITS`训练的语音合成（TTS）模型，支持中、英、日等多种语言

## Install
```shell
brew install portaudio # for mac
pip install parrots
```
or
```shell
git clone https://github.com/shibing624/parrots.git
cd parrots
python setup.py install
```

## Demo
Official Demo: https://www.mulanai.com/product/asr/

## Usage
### ASR
example: [examples/demo_asr.py](examples/demo_asr.py)
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
example: [examples/demo_tts.py](examples/demo_tts.py)
```python
import sys

sys.path.append('..')
from parrots import TextToSpeech

if __name__ == '__main__':
    m = TextToSpeech()
    # say text
    m.speak('北京图书馆')

    # generate wav file to path
    m.synthesize('北京图书馆', output_wav_path='./out.wav')
```

output:
```
北京图书馆
```

## Dataset

## 语音库
从SourceForge下载语音库[`syllables.zip`](https://sourceforge.net/projects/hantts/files/?source=navbar)
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
  year={2022},
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
#### ASR
- [EAT: Enhanced ASR-TTS for Self-supervised Speech Recognition](https://arxiv.org/abs/2104.07474)
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
- [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
#### TTS(Speech Synthesis)
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- [keonlee9420/Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2)
- [TensorSpeech/TensorflowTTS](https://github.com/TensorSpeech/TensorflowTTS)

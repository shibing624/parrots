# parrots
parrots, Automatic Speech Recognition(ASR), Text-To-Speech(TTS) engine.


## Install
```
brew install portaudio
pip3 install -r requirements.txt
```

* pip3 install parrots
* Or
```
git clone https://github.com/shibing624/parrots.git
cd parrots
python3 setup.py install
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
from parrots import SpeechRecognition, Pinyin2Hanzi

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    m = SpeechRecognition()
    r = m.recognize_speech_from_file(os.path.join(pwd_path, 'tushuguan.wav'))
    print('[提示] 语音识别结果：', r)

    n = Pinyin2Hanzi()
    text = n.pinyin_2_hanzi(r)
    print('[提示] 语音转文字结果：', text)

```

output:
```
北京图书馆
```

### TTS
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

### 语音库
从SourceForge下载语音库[`syllables.zip`](https://sourceforge.net/projects/hantts/files/?source=navbar)，并解压到`parrots/data`目录下

```shell
wget https://sourceforge.net/projects/hantts/files/syllables.zip --no-check-certificate
```

### 录制新的语音库
- 按阴平、阳平、上声、去声、轻声的顺序录下 mapping.json 里每一个音节的五个声调
- 按开头字母(letter)分组, 将文件存在 ./recording/{letter}.wav下
- 运行 `python process.py {letter}` 将{letter}.wav 完整的录音分成独立的拼音
- 检查核对`./pre`文件夹中的拼音.wav后导入文件夹`./syllables`


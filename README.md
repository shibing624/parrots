# parrots
parrots, Automatic Speech Recognition(ASR), Text-To-Speech(TTS) engine.


## install
```
brew install portaudio
pip3 install -r requirements.txt
```

这里请注意，python37不支持pyaudio的pip安装，直接下载whl文件，然后pip安装，参考[这里](https://blog.csdn.net/COCO56/article/details/104190090)

* pip3 install parrots
* Or
```
git clone https://github.com/shibing624/parrots.git
cd parrots
python3 setup.py install
```


## usage
### speech recognition
input:
```
import parrots

text = parrots.speech_recognition_from_file('./16k.wav')
print(text)

```

output:
```
北京图书馆
```

### tts
input:
```
import parrots

audio_file_path = parrots.synthesize('北京图书馆')
print(audio_file_path)

```

output:
```
北京图书馆 语音文件路径
```

## update

### 语音库
从SourceForge下载语音库[`syllables.zip`](https://sourceforge.net/projects/hantts/files/?source=navbar)，并解压到`parrots/data`目录下

### 录制新的语音库
- 按阴平、阳平、上声、去声、轻声的顺序录下 mapping.json 里每一个音节的五个声调
- 按开头字母(letter)分组, 将文件存在 ./recording/{letter}.wav下
- 运行 `python process.py {letter}` 将{letter}.wav 完整的录音分成独立的拼音
- 检查核对`./pre`文件夹中的拼音.wav后导入文件夹`./syllables`


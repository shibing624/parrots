# parrots
parrots, Automatic Speech Recognition(ASR), Text-To-Speech(TTS) engine.


## install
```
brew install portaudio
pip3 install -r requirements.txt
```

这里请注意，python37不支持pyaudio的pip安装，直接下载whl文件，然后pip安装，参考[这里](https://blog.csdn.net/COCO56/article/details/104190090)

如果安装失败：

sudo apt install python3-dev -y
sudo apt-get install portaudio19-dev python-all-dev



* pip3 install parrots
* Or
```
git clone https://github.com/shibing624/parrots.git
cd parrots
python3 setup.py install
```


## usage
### speech recognition
打开 "tests/speech_reco_test.py"，运行



output:
```
北津科技过了(北京科技馆)
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


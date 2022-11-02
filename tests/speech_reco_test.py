# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from parrots import SpeechRecognition
from parrots.pinyin2hanzi import Pinyin2Hanzi

ms = SpeechRecognition(pinyin_path='../parrots/data/pinyin_hanzi_dict.txt',
                       model_path='../parrots/data/speech_model/speech_recognition.model')
r = ms.recognize_speech_from_file('../parrots/data/16k.wav')
print('[提示] 语音识别结果：\n', r)

p2h = Pinyin2Hanzi(model_dir='../parrots/data/pinyin2hanzi')

text = p2h.pinyin_2_hanzi(r)
print('语音转文字结果：\n', text)

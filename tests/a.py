# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


from parrots.speech_recognition import SpeechRecognition
from parrots.pinyin2hanzi import Pinyin2Hanzi

ms = SpeechRecognition()
r = ms.recognize_speech_from_file('../parrots/data/16k.wav')
print('[提示] 语音识别结果：\n', r)

p2h = Pinyin2Hanzi()

text = p2h.pinyin_2_text(r)
print('语音转文字结果：\n', text)

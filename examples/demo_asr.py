# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('..')
from parrots import SpeechRecognition, Pinyin2Hanzi

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    m = SpeechRecognition()
    r = m.recognize_speech_from_file(os.path.join(pwd_path, '16k.wav'))
    print('[提示] 语音识别结果：', r)

    n = Pinyin2Hanzi()
    text = n.pinyin_2_hanzi(r)
    print('[提示] 语音转文字结果：', text)

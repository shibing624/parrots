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


def test1():
    m = SpeechRecognition()
    r = m.recognize_speech_from_file(os.path.join(pwd_path, '../examples/tushuguan.wav'))
    print('[提示] 语音识别结果：\n', r)

    p2h = Pinyin2Hanzi()
    text = p2h.pinyin_2_hanzi(r)
    print('语音转文字结果：\n', text)

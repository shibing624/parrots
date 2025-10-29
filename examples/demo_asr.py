# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('..')
from parrots import SpeechRecognition

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    m = SpeechRecognition(model_name_or_path='BELLE-2/Belle-distilwhisper-large-v2-zh')
    r = m.recognize_speech_from_file(os.path.join(pwd_path, 'tushuguan.wav'))
    print('语音识别结果：', r)

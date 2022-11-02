# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from parrots import TextToSpeech

def test1():
    tts = TextToSpeech(syllables_dir='../parrots/data/syllables')
    print('中文语言合成')
    tts.synthesize(input_text=input('输入中文：'), output_wav_path='./out.wav')

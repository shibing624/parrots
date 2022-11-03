# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 中文语言合成
"""
import os
import sys

sys.path.append('..')
from parrots import TextToSpeech


def test1():
    tts = TextToSpeech(syllables_dir='../parrots/data/syllables')
    tts.synthesize(input_text='活雷锋在北京', output_wav_path='./out.wav')

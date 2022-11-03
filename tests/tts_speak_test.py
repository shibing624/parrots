# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from parrots.tts import TextToSpeech


def test1():
    tts = TextToSpeech(syllables_dir='../parrots/data/syllables')
    tts.speak('你好，我是小明，我来自中国。')

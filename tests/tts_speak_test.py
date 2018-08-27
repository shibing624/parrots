# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from parrots.tts import TextToSpeech

if __name__ == '__main__':
    tts = TextToSpeech(syllables_dir='../parrots/data/syllables')
    while True:
        tts.speak(input('输入中文：\n'))

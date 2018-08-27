# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from parrots.tts import TextToSpeech

if __name__ == '__main__':
    tts = TextToSpeech(syllables_dir='../parrots/data/syllables')
    print('中文语言合成')
    while True:
        tts.synthesize(input_text=input('输入中文：'), output_wav_path='./out.wav')

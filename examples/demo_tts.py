# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from parrots import TextToSpeech

if __name__ == '__main__':
    m = TextToSpeech()
    # say text
    m.speak('北京图书馆')

    # generate wav file to path
    m.synthesize('北京图书馆', output_wav_path='./out.wav')

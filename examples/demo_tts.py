# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import parrots

if __name__ == '__main__':
    # say text
    parrots.speak('北京图书馆')

    # generate wav file to path
    parrots.synthesize('北京图书馆', output_wav_path='./out.wav')

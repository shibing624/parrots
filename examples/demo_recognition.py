# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import parrots

r = parrots.recognize_speech_from_file('../parrots/data/16k.wav')
print('[提示] 语音识别结果：\n', r)

text = parrots.pinyin_2_hanzi(r)
print('[提示] 语音转文字结果：\n', text)

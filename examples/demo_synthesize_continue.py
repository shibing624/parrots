# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import parrots

# generate wav file to path
texts = ['北京图书馆',
         '你是好人吗？232个人是坏人',
         '年轻人的社区，3.14是圆周率',
         ]

# say text
for i in texts:
    parrots.synthesize(i)

parrots.synthesize('北京图书馆', output_wav_path='a.wav')
parrots.synthesize('天津社会图书店')

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import parrots

texts = ['北京图书馆',
         '你是好人吗？232个人是坏人',
         '你好北京，我爱你北京',
         ]

# say text
for i in texts:
    parrots.speak(i)

parrots.speak('天津社会图书店')

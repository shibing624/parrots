# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

pinyin2hanzi_dir = os.path.join(pwd_path, 'data/pinyin2hanzi')
pinyin_hanzi_dict_path = os.path.join(pwd_path, 'data/pinyin2hanzi/pinyin_hanzi_dict.txt')
speech_recognition_model_path = os.path.join(pwd_path, 'data/speech_model/speech_recognition.model')
syllables_dir = os.path.join(pwd_path, 'data/syllables')

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from .pinyin2hanzi import Pinyin2Hanzi
from .speech_recognition import SpeechRecognition
from .tts import TextToSpeech

__version__ = '0.1.5'

sr = SpeechRecognition()
recognize_speech_from_file = sr.recognize_speech_from_file

p2h = Pinyin2Hanzi()
pinyin_2_hanzi = p2h.pinyin_2_hanzi

t2s = TextToSpeech()
update_syllables_dir = t2s.update_syllables_dir
speak = t2s.speak
synthesize = t2s.synthesize

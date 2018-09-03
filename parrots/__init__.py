# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

from .pinyin2hanzi import Pinyin2Hanzi
from .speech_recognition import SpeechRecognition
from .tts import TextToSpeech

__version__ = '0.1.3'
_pwd_path = os.path.abspath(os.path.dirname(__file__))
_get_abs_path = lambda path: os.path.normpath(os.path.join(_pwd_path, path))

DEFAULT_PINYIN_DICT = _get_abs_path('data/pinyin_hanzi_dict.txt')
DEFAULT_SPEECH_MODEL = _get_abs_path('data/speech_model/speech_recognition.model')
DEFAULT_PINYIN2HANZI_DIR = _get_abs_path('data/pinyin2hanzi')
DEFAULT_SYLLABLES_DIR = _get_abs_path('data/syllables')

sr = SpeechRecognition(pinyin_path=DEFAULT_PINYIN_DICT,
                       model_path=DEFAULT_SPEECH_MODEL)
recognize_speech_from_file = sr.recognize_speech_from_file

p2h = Pinyin2Hanzi(model_dir=DEFAULT_PINYIN2HANZI_DIR)
pinyin_2_hanzi = p2h.pinyin_2_hanzi

t2s = TextToSpeech(syllables_dir=DEFAULT_SYLLABLES_DIR)
update_syllables_dir = t2s.update_syllables_dir
speak = t2s.speak
synthesize = t2s.synthesize

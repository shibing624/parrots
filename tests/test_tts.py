# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 中文语言合成
"""

import os
import sys
import unittest

sys.path.append('..')
from parrots.tts import TextToSpeech


class TestTTS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pwd_path = os.path.abspath(os.path.dirname(__file__))
        cls.m = TextToSpeech()

    def test_tts_to_file(self):
        self.m.predict(text='活雷锋在北京', output_path='./out.wav')
        self.assertEqual(os.path.exists('./out.wav'), True)

    def test_tts_to_audio(self):
        audio_output = self.m.predict('你好，我是小明，我来自中国。')
        self.assertEqual(len(audio_output) > 0, True)


if __name__ == '__main__':
    unittest.main()

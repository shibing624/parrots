# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys
import unittest

sys.path.append('..')
from parrots import SpeechRecognition


class TestSpeechRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pwd_path = os.path.abspath(os.path.dirname(__file__))

    def test_recognize_speech_from_file(self):
        # 创建 SpeechRecognition 的实例
        m = SpeechRecognition()
        file_path = os.path.join(self.pwd_path, '..', 'examples', 'tushuguan.wav')
        result = m.recognize_speech_from_file(file_path)
        res = result.get("text")
        self.assertEqual(res, '北京图书馆')


if __name__ == '__main__':
    unittest.main()

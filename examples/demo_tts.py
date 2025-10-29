# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
from parrots.tts import TextToSpeech
from parrots.log import set_log_level

set_log_level("DEBUG")

if __name__ == "__main__":
    m = TextToSpeech(
        speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
        speaker_name="MaiMai",
        device='cpu'
    )
    m.predict(
        text='你好，欢迎来到北京。这是一个合成录音文件的演示。Welcome to Beijing!',
        output_path="output_audio.wav"
    )

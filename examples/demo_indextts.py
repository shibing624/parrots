# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from parrots.indextts.inference import IndexTTS2
from parrots.log import set_log_level

set_log_level("DEBUG")

tts = IndexTTS2()
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(speak_reference_audio_path='wav/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True,
          use_random=False, verbose=True)

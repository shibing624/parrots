# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from parrots.indextts.inference import IndexTTS2

tts = IndexTTS2()
text = "他好帅啊，我好喜欢他！我们一起吃吃喝喝"
tts.infer(speak_reference_audio_path_or_name='wav/voice_12.wav', text=text, output_path="happy.wav", emo_alpha=0.6, use_emo_text=True,
          use_random=False, verbose=True)

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from parrots.indextts.infer import IndexTTS2

tts = IndexTTS2(model_dir="/apdcephfs_qy3/share_7435715/data/models/IndexTeam--IndexTTS-2")
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='wav/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True,
          use_random=False, verbose=True)

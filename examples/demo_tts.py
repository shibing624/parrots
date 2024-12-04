# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
import parrots
from parrots.tts import TextToSpeech

parrots_path = parrots.__path__[0]
sys.path.append(parrots_path)  # add parrots to sys.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_model", type=str, default="shibing624/parrots-gpt-sovits-speaker-maimai",
                        help="Model path")
    parser.add_argument("--speaker_name", type=str, default="MaiMai", help="Name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--half", action="store_true", help="Use half precision instead of float32")
    parser.add_argument("--text", type=str, default="你好，欢迎来北京。welcome to the city.", help="input text")
    parser.add_argument("--lang", type=str, default="auto", help="Language of the text, zh, en, jp, auto")
    parser.add_argument("--output_path", type=str, default="output_audio.wav", help="output wav")
    args = parser.parse_args()
    print(f"args: {args}")
    m = TextToSpeech(
        speaker_model_path=args.speaker_model,
        speaker_name=args.speaker_name,
        device=args.device,
        half=args.half
    )
    m.predict(
        text=args.text,
        text_language=args.lang,
        output_path=args.output_path
    )

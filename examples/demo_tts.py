# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
from parrots.tts import TextToSpeech

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert",
        type=str,
        default="pretrained_models/chinese-roberta-wwm-ext-large",
        help="Path to the pretrained BERT model",
    )
    parser.add_argument(
        "--hubert",
        type=str,
        default="pretrained_models/chinese-hubert-base",
        help="Path to the pretrained HuBERT model",
    )
    parser.add_argument(
        "--sovits",
        type=str,
        # default="pretrained_models/s2G488k.pth",
        default="my_models/xiaowu_e12_s108.pth",
        help="Path to the pretrained SoVITS",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        # default="pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        default="my_models/xiaowu-e10.ckpt",
        help="Path to the pretrained GPT",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--half", action="store_true", help="Use half precision instead of float32")
    parser.add_argument("--text", type=str, default="你好，欢迎来北京。welcome to the city.", help="input text")
    parser.add_argument("--lang", type=str, default="zh", help="Language of the text, zh, en, jp, auto")
    parser.add_argument("--ref_wav_path", type=str, default="./ref.wav", help="reference wav")
    parser.add_argument("--ref_text", type=str,
                        default="大家好，我是宁宁。我中文还不是很熟练，但是希望大家能喜欢我的声音，喵喵喵！",
                        help="reference text")
    parser.add_argument("--ref_lang", type=str, default="zh", help="reference wav language")
    parser.add_argument("--output_path", type=str, default="output_audio.wav", help="output wav")
    args = parser.parse_args()
    print(f"args: {args}")
    m = TextToSpeech(
        bert_model_path=args.bert,
        hubert_model_path=args.hubert,
        sovits_model_path=args.sovits,
        gpt_model_path=args.gpt,
        device=args.device,
        half=args.half
    )
    m.predict(
        ref_wav_path=args.ref_wav_path,
        ref_prompt=args.ref_text,
        ref_language=args.ref_lang,
        text=args.text,
        text_language=args.lang,
        output_path=args.output_path
    )

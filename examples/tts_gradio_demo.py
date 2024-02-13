# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import hashlib
import os
import ssl

import gradio as gr
import torch
from loguru import logger

ssl._create_default_https_context = ssl._create_unverified_context
import nltk

nltk.download('cmudict')
from parrots import TextToSpeech

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device: {device}")
half = True if device == "cuda" else False

m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
    device=device,
    half=half
)
m.predict(
    text="你好，欢迎来北京。welcome to the city.",
    text_language="auto",
    output_path="output_audio.wav"
)
assert os.path.exists("output_audio.wav"), "output_audio.wav not found"


def get_text_hash(text: str):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def do_tts_wav_predict(text: str, output_path: str = None):
    if output_path is None:
        output_path = f"output_audio_{get_text_hash(text)}.wav"
    if not os.path.exists(output_path):
        m.predict(text, text_language="auto", output_path=output_path)
    return output_path


with gr.Blocks(title="parrots WebUI") as app:
    gr.Markdown(value="""
    # <center>在线语音生成（parrots）speaker:主播卖卖\n

    ### <center>parrots项目：https://github.com/shibing624/parrots\n
    ### <center>数据集下载：https://huggingface.co/datasets/XzJosh/audiodataset\n
    ### <center>声音归属：扇宝 https://space.bilibili.com/698438232\n
    ### <center>模型训练：https://github.com/RVC-Boss/GPT-SoVITS\n
    ### <center>使用本模型请严格遵守法律法规！发布二创作品请标注本项目作者及链接、作品使用GPT-SoVITS AI生成！\n
    ### <center>⚠️在线端不稳定且生成速度较慢，建议使用parrots本地推理！\n
                """)

    with gr.Group():
        gr.Markdown(value="*请填写需要语音合成的文本")
        with gr.Row():
            text = gr.Textbox(label="需要合成的文本(建议100字以内)", value="", placeholder="请输入短文本", lines=3)
            inference_button = gr.Button("合成语音", variant="primary")
            output = gr.Audio(label="输出的语音")
        inference_button.click(
            do_tts_wav_predict,
            [text],
            [output],
        )

app.queue(max_size=10)
app.launch(inbrowser=True, debug=True, show_api=True, share=True)

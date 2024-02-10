# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import re
import struct
import sys
import wave
from typing import Dict, Any

import ffmpeg
import librosa
import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.append('..')
from parrots import cnhubert
from parrots.mel_processing import spectrogram_torch
from parrots.synthesizer_model import SynthesizerModel
from parrots.t2s_module import Text2SemanticLightningModule, Text2SemanticDecoder
from parrots.text_utils import clean_text, cleaned_text_to_sequence

model_mappings: Dict[str, Any] = {}
splits_flags = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def to_device(model, device=None):
    "A helper function to move a model to GPU"
    if args.half:
        model = model.half()
    if device:
        return model.to(device)
    else:
        return model.to(args.device)


def get_bert_feature(text, word2ph):
    bert = model_mappings["bert"]
    tokenizer, bert_model = bert["tokenizer"], bert["model"]

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(bert_model.device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def inference(ref_wav_path, prompt_text, prompt_language, text, text_language):
    hps = model_mappings["hps"]

    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if args.half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = to_device(torch.from_numpy(wav16k))
        zero_wav_torch = to_device(torch.from_numpy(zero_wav))
        wav16k = torch.cat([wav16k, zero_wav_torch])

        ssl_model = model_mappings["ssl_model"]
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        vq_model = model_mappings["vq_model"]
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

    # step1
    if prompt_language == "en":
        phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
    else:
        phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)

    text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
    if text[-1] not in splits_flags:
        text += "。" if text_language != "en" else "."
    texts = text.split("\n")
    logger.debug(f"texts: {texts}")

    # audio_opt = []
    if prompt_language == "en":
        bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
    else:
        bert1 = nonen_get_bert_inf(prompt_text, prompt_language)

    for text in texts:
        if len(text.strip()) == 0:
            continue
        if text_language == "en":
            phones2, word2ph2, norm_text2 = clean_text_inf(text, text_language)
        else:
            phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text, text_language)

        if text_language == "en":
            bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
        else:
            bert2 = nonen_get_bert_inf(text, text_language)

        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(args.device).unsqueeze(0)
        bert = bert.to(args.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(args.device)
        prompt = prompt_semantic.unsqueeze(0).to(args.device)

        # step2
        hz = 50
        t2s = model_mappings["t2s"]
        t2s_model, t2s_topk, t2s_max_sec = t2s["model"], t2s["top_k"], t2s["max_sec"]
        with torch.no_grad():
            # pred_semantic, idx = t2s_model.model.infer_panel(
            pred_semantic, idx = t2s_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=t2s_topk,
                early_stop_num=hz * t2s_max_sec,
            )

        # step3
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0) #mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)
        refer = to_device(refer)

        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(args.device).unsqueeze(0),
                refer,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  # 试试重建不带上prompt部分

        audio_raw = (np.concatenate([audio, zero_wav], 0) * 32768).astype(np.int16)
        yield np.int16(audio_raw / np.max(np.abs(audio_raw)) * 32767).tobytes()


def splite_en_inf(sentence, language):
    pattern = re.compile(r"[a-zA-Z. ]+")
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)

    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(args.device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if args.half == True else torch.float32,
        ).to(args.device)

    return bert


def nonen_clean_text_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "en" or "ja":
            pass
        else:
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    # print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = " ".join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    # print(textlist)
    # print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def load_models():
    # BERT
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    model = AutoModelForMaskedLM.from_pretrained(args.bert)
    model_mappings["bert"] = {"tokenizer": tokenizer, "model": to_device(model)}

    # HuBERT
    model_mappings["ssl_model"] = to_device(cnhubert.get_model(args.hubert))

    # SoVITS
    dict_s2 = torch.load(args.sovits, map_location="cpu")
    hps = DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    model_mappings["hps"] = hps
    logger.info(f"hps: {hps}")
    logger.info(f"hps.model: {hps.model}")

    # VQ
    vq_model = SynthesizerModel(
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    vq_model = to_device(vq_model)
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    model_mappings["vq_model"] = vq_model
    logger.info(f"vq_model: {vq_model}")

    # GPT
    dict_s1 = torch.load(args.gpt, map_location="cpu")
    config = dict_s1["config"]
    logger.info(f"config: {config}")
    # t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model = Text2SemanticDecoder(config)
    t2s_model.load_state_dict(dict_s1["weight"], strict=False)
    t2s_model = to_device(t2s_model)
    t2s_model.eval()
    logger.info(f"t2s_model: {t2s_model}")
    model_mappings["t2s"] = {
        "model": t2s_model,
        "top_k": config["inference"]["top_k"],
        "max_sec": config["data"]["max_sec"],
    }
    total = sum([param.nelement() for param in t2s_model.parameters()])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))


def pcm16_header(rate, size=1000000000, channels=1):
    # Header for 16-bit PCM, modify from scipy.io.wavfile.write
    fs = rate
    # size = data.nbytes  # length * sizeof(nint16)

    header_data = b"RIFF"
    header_data += struct.pack("i", size + 44)
    header_data += b"WAVE"

    # fmt chunk
    header_data += b"fmt "
    format_tag = 1  # PCM
    bit_depth = 2 * 8  # 2 bytes
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)
    fmt_chunk_data = struct.pack("<HHIIHH", format_tag, channels, fs, bytes_per_second, block_align, bit_depth)

    header_data += struct.pack("<I", len(fmt_chunk_data))
    header_data += fmt_chunk_data
    header_data += b"data"
    return header_data + struct.pack("<I", size)


def save_wav(wav_data, filename, sample_rate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位样本，因此采样宽度为2字节
        wf.setframerate(sample_rate)  # 设置采样率
        wf.writeframes(wav_data)


class TextToSpeech:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS")
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
    parser.add_argument("--lang", type=str, default="zh", help="Language of the text, zh, en, jp")
    parser.add_argument("--ref_wav_path", type=str, default="../examples/ref.wav", help="reference wav")
    parser.add_argument("--ref_text", type=str,
                        default="大家好，我是宁宁。我中文还不是很熟练，但是希望大家能喜欢我的声音，喵喵喵！",
                        help="reference text")
    parser.add_argument("--ref_lang", type=str, default="zh", help="reference wav language")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    load_models()
    hps = model_mappings["hps"]

    params = {
        "ref_wav_path": args.ref_wav_path,
        "prompt_text": args.ref_text,
        "prompt_language": args.ref_lang,
        "text": args.text,
        "text_language": args.lang,
    }
    logger.info(params)
    sampling_rate = hps.data.sampling_rate
    audio_stream = inference(**params)
    # 先返回PCM的头部，将音频长度设置成较大的值以便后面分块发送音频数据
    pcm_header_data = pcm16_header(sampling_rate)
    audio_data = b''.join([audio_chunk for audio_chunk in audio_stream])
    # 获取WAV文件的头部和音频数据，重新合并PCM头部和音频数据
    complete_audio = pcm_header_data + audio_data
    output_wav_path = 'output_audio.wav'
    save_wav(complete_audio, output_wav_path, sample_rate=sampling_rate)
    logger.info(f"Saved to {output_wav_path}")

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
import re
import struct
import sys
import wave
from enum import Enum
from typing import Union, Optional

import librosa
import numpy as np
import soundfile
import torch
import nltk
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from parrots.log import logger
try:
    import parrots.utils as _parrots_utils
    sys.modules.setdefault('utils', _parrots_utils)
except Exception as e:
    logger.debug(f"Failed to alias utils -> parrots.utils: {e}")

from parrots import cnhubert
from parrots.mel_processing import spectrogram_torch
from parrots.synthesizer_model import SynthesizerModel
from parrots.t2s_model import Text2SemanticDecoder
from parrots.text_utils import clean_text, cleaned_text_to_sequence
from parrots.symbols import sentence_split_symbols


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

# Constants for speaker model
CONFIG_NAME = "config.json"
SOVITS_MODEL_NAME = "sovits.pth"
GPT_MODEL_NAME = "gpt.ckpt"
REF_WAV_NAME = "ref.wav"
SPEAKER_NAMES = ["MaiMai", "XingTong", "XuanShen", "KusanagiNene", "LongShouRen", "KuileBlanc"]


class LANG(Enum):
    auto = 0  # 多语种按句子切分识别各自语种
    zh = 1  # 全部按中文识别
    en = 2  # 全部按英文识别
    ja = 3  # 全部按日文识别

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return LANG[s]
        except KeyError:
            raise ValueError()


def load_audio(file, sr):
    """
    Load audio file and resample to target sample rate.
    First try using ffmpeg, if failed, fallback to librosa or soundfile.
    
    Args:
        file: path to audio file
        sr: target sample rate
    
    Returns:
        audio: numpy array of audio samples
    """
    file = (
        file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )  # 防止小白拷路径头尾带了空格和"和回车
    
    # Try ffmpeg first (fastest and most robust)
    try:
        import ffmpeg
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.float32).flatten()
    except Exception as e:
        logger.debug(f"Failed to load audio with ffmpeg: {e}, trying librosa to load.")
    
    # Fallback 1: Try librosa (most common audio library)
    try:
        audio, orig_sr = librosa.load(file, sr=sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load audio with librosa: {e}, trying soundfile...")
    
    # Fallback 2: Try soundfile
    try:
        audio, orig_sr = soundfile.read(file, dtype='float32')
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # Resample if necessary
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr, res_type='kaiser_best')
        logger.info(f"Successfully loaded audio with soundfile (fallback method)")
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to load audio with soundfile: {e}")
    
    # All methods failed
    raise RuntimeError(f"Failed to load audio file {file} with all available methods (ffmpeg, librosa, soundfile)")


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


def get_language_texts(text):
    """
    Automatically detect and split text by language (zh, en, ja).
    Returns a list of dicts with 'lang' and 'text' keys.
    """
    # Define character ranges for different languages
    # Chinese: \u4e00-\u9fff
    # Japanese Hiragana: \u3040-\u309f
    # Japanese Katakana: \u30a0-\u30ff
    # English and numbers: a-zA-Z0-9
    pattern = re.compile(
        r'([\u4e00-\u9fff]+|[\u3040-\u309f\u30a0-\u30ff]+|[a-zA-Z0-9\s]+|[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9\s]+)'
    )
    
    results = []
    for match in pattern.finditer(text):
        segment = match.group(0)
        if not segment.strip():
            continue
            
        # Detect language
        if re.search(r'[\u4e00-\u9fff]', segment):
            lang = 'zh'
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', segment):
            lang = 'ja'
        elif re.search(r'[a-zA-Z]', segment):
            lang = 'en'
        else:
            # For punctuation and other characters, merge with previous segment
            if results:
                results[-1]['text'] += segment
            continue
        
        # Merge with previous segment if same language
        if results and results[-1]['lang'] == lang:
            results[-1]['text'] += segment
        else:
            results.append({'lang': lang, 'text': segment})
    
    return results


def split_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
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
    # Merge punctuation into previous word
    for i in range(len(textlist) - 1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i - 1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i + 1]:
            textlist[i] += textlist[i + 1]
            del textlist[i + 1]
            del langlist[i + 1]
        else:
            i += 1

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


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


def load_json_file(filepath):
    """
    加载指定路径的JSON文件，并返回其内容。
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def merge_short_text_in_array(texts, threshold):
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in sentence_split_symbols) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def sentence_split_by_symbol(input_text):
    input_text = input_text.replace("……", "。").replace("——", "，")
    if input_text[-1] not in sentence_split_symbols:
        input_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(input_text)
    split_sents = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if input_text[i_split_head] in sentence_split_symbols:
            i_split_head += 1
            split_sents.append(input_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return split_sents


def sentence_split_by_length(input_text, max_len=50):
    """每50个字符切分一次"""
    input_text = input_text.strip("\n")
    sentences = sentence_split_by_symbol(input_text)
    if len(sentences) < 2:
        return input_text
    sents = []
    summ = 0
    tmp_str = ""
    for i in range(len(sentences)):
        summ += len(sentences[i])
        tmp_str += sentences[i]
        if summ > max_len:
            summ = 0
            sents.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        sents.append(tmp_str)
    if len(sents) > 1 and len(sents[-1]) < max_len:  # 如果最后一个太短了，和前一个合一起
        sents[-2] = sents[-2] + sents[-1]
        sents = sents[:-1]
    return "\n".join(sents)


class TextToSpeech:
    def __init__(
            self,
            bert_model_path: str = "shibing624/parrots-chinese-roberta-wwm-ext-large",
            hubert_model_path: str = "shibing624/parrots-chinese-hubert-base",
            sovits_model_path: str = None,
            gpt_model_path: str = None,
            speaker_model_path: str = "shibing624/parrots-gpt-sovits-speaker-maimai",
            speaker_name: Optional[str] = "MaiMai",
            device: Optional[str] = None,
            half: Optional[bool] = False,
            ensure_nltk: bool = True,
    ):
        """
        Args:
            bert_model_path: str, path to the pretrained BERT model
            hubert_model_path: str, path to the pretrained HuBERT model
            sovits_model_path: str, path to the pretrained SoVITS, if None, use the speaker_model_path
            gpt_model_path: str, path to the pretrained GPT, if None, use the speaker_model_path
            speaker_model_path: str, path to the pretrained speaker model,
                if sovits_model_path and gpt_model_path are None, use this model, else ignore it
            speaker_name: str, name of the speaker, default is "MaiMai"
            device: str, device to run on, "cuda", "cpu" or "mps"
            half: bool, use half precision instead of float32
            ensure_nltk: bool, whether to check/download NLTK resources during initialization
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "cpu"
            else:
                device = "cpu"
        self.device = device
        logger.debug("Use device: {}".format(self.device))
        self.half = half
        self.dtype = torch.float16 if half else torch.float32

        # Ensure required NLTK resources are available (optional to avoid import-time side effects)
        if ensure_nltk:
            self._ensure_nltk_data()

        # BERT
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_path)
        self.bert_model = self.to_device(self.bert_model)

        # Speaker
        if bool(sovits_model_path) != bool(gpt_model_path):  # Check if only one of them is provided
            raise ValueError("sovits_model_path and gpt_model_path must be provided together")
        if sovits_model_path is None and gpt_model_path is None:
            if speaker_model_path:
                logger.info("Load pretrained parrots speaker: {}".format(speaker_model_path))
                if os.path.exists(speaker_model_path):
                    # Load from path
                    model_path = speaker_model_path
                else:
                    # Load from huggingface model hub
                    model_path = snapshot_download(speaker_model_path)
                sovits_model_path = os.path.join(model_path, speaker_name, SOVITS_MODEL_NAME)
                gpt_model_path = os.path.join(model_path, speaker_name, GPT_MODEL_NAME)
                # Set ref_wav_path to the speaker's reference wav file
                config_file = os.path.join(model_path, speaker_name, CONFIG_NAME)
                if os.path.exists(config_file):
                    ref_config = load_json_file(config_file)
                    logger.debug(f"Reference speaker config: {ref_config}, loaded from {config_file}")
                    self.ref_wav_path = os.path.join(model_path, speaker_name, REF_WAV_NAME)
                    self.ref_prompt = ref_config.get("reference_prompt", "")
                    self.ref_language = ref_config.get("reference_language", "")
                else:
                    raise ValueError(f"Config file not found: {config_file}")
            else:
                raise ValueError("sovits_model_path, gpt_model_path or speaker_model_path must be provided")

        # SoVITS
        sovits_dict = torch.load(sovits_model_path, map_location="cpu", weights_only=False)
        hps = DictToAttrRecursive(sovits_dict["config"])
        logger.debug(f"SoVITS config: {hps}")
        vq_model = SynthesizerModel(
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        self.sampling_rate = hps.data.sampling_rate
        self.hps = hps
        vq_model = self.to_device(vq_model)
        vq_model.eval()
        vq_model.load_state_dict(sovits_dict["weight"], strict=False)
        self.vq_model = vq_model

        # GPT
        gpt_dict = torch.load(gpt_model_path, map_location="cpu", weights_only=False)
        config = gpt_dict["config"]
        logger.debug(f"GPT config: {config}")
        t2s_model = Text2SemanticDecoder(config)
        # Convert Text2SemanticLightningModule to Text2SemanticDecoder model
        adjusted_state_dict = self.adjust_keys(gpt_dict["weight"])
        t2s_model.load_state_dict(adjusted_state_dict)
        t2s_model = self.to_device(t2s_model)
        t2s_model.eval()
        self.t2s_model = t2s_model
        self.t2s_max_sec = config["data"]["max_sec"]
        total = sum([param.nelement() for param in t2s_model.parameters()])
        logger.debug("Number of t2s model parameter: %.2fM" % (total / 1e6))

        # HuBERT
        self.ssl_model = self.to_device(cnhubert.get_model(hubert_model_path, self.sampling_rate))
        
        # Cache for streaming TTS
        self._ref_cache = {}

    def _ensure_nltk_data(self):
        """Ensure NLTK tagger resources exist, download quietly if missing.
        Moved from module scope to avoid side effects at import time for PyPI.
        """
        try:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                logger.info("Downloading NLTK averaged_perceptron_tagger_eng...")
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                logger.info("Downloading NLTK averaged_perceptron_tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to ensure NLTK data: {e}")

    @staticmethod
    def adjust_keys(state_dict):
        new_state_dict = {}
        # 对于每一个在状态字典中的键和值
        for k, v in state_dict.items():
            # 如果键以"model."开始，就去掉这个前缀
            if k.startswith("model."):
                new_key = k[len("model."):]  # 删除前缀
            else:
                new_key = k
            new_state_dict[new_key] = v  # 加入调整后的键值对到新字典
        return new_state_dict

    def to_device(self, model):
        """A helper function to move a model to GPU or cpu"""
        if self.half:
            model = model.half()
        return model.to(self.device)

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.bert_tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.bert_model.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)), dtype=torch.float16 if self.half else torch.float32,
            ).to(self.device)
        return bert

    def nonen_get_bert_inf(self, text, language):
        if language == "auto":
            textlist = []
            langlist = []
            for tmp in get_language_texts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            textlist, langlist = split_en_inf(text, language)
        bert_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert_inf = self.get_bert_inf(phones, word2ph, norm_text, lang)
            bert_list.append(bert_inf)
        return torch.cat(bert_list, dim=1)

    def nonen_clean_text_inf(self, text, language):
        if language == "auto":
            textlist = []
            langlist = []
            for tmp in get_language_texts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            textlist, langlist = split_en_inf(text, language)
        logger.debug(textlist)
        logger.debug(langlist)
        phones_list = []
        word2ph_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            phones_list.append(phones)
            if lang == "zh":
                word2ph_list.append(word2ph)
            norm_text_list.append(norm_text)
        phones = sum(phones_list, [])
        word2ph = sum(word2ph_list, [])
        norm_text = ' '.join(norm_text_list)
        return phones, word2ph, norm_text

    def get_cleaned_text_final(self, text, language):
        if language in {"en", "zh", "ja"}:
            phones, word2ph, norm_text = clean_text_inf(text, language)
        elif language in {"auto"}:
            phones, word2ph, norm_text = self.nonen_clean_text_inf(text, language)
        else:
            raise ValueError(f"Unsupported language: {language}")
        return phones, word2ph, norm_text

    def get_bert_final(self, phones, word2ph, text, language):
        if language == "en":
            bert = self.get_bert_inf(phones, word2ph, text, language)
        elif language in {"ja", "auto"}:
            bert = self.nonen_get_bert_inf(text, language)
        elif language == "zh":
            bert = self.get_bert_feature(text, word2ph).to(self.device)
        else:
            bert = torch.zeros((1024, len(phones))).to(self.device)
        return bert

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @torch.inference_mode()
    def predict(
            self,
            text: str,
            text_language: Union[str, LANG] = "auto",
            speed: float = 1.0,
            output_path: Optional[str] = None,
            ref_wav_path: str = None,
            ref_prompt: str = None,
            ref_language: Union[str, LANG] = None,
            top_k: int = 20,
            top_p: float = 0.6,
            temperature: float = 0.6,
    ):
        """
        Args:
            text: str, target text 要语音合成的文本
            text_language: str, language of the target text 要语音合成的文本的语种
            speed: float, speed of speech 语速
            output_path: str, path to save the output wav file 保存语音合成结果的路径，可选
            ref_wav_path: str, path to the reference wav file 参考音频文件
            ref_prompt: str, reference prompt 参考音频对应的文本
            ref_language: str, language of the reference prompt 参考音频对应的文本的语种
            top_k: int, top k
            top_p: float, top p
            temperature: float, temperature
        Returns:
            audio array: generator, audio stream, numpy array
        """
        if ref_wav_path is None:
            ref_wav_path = self.ref_wav_path
        if ref_prompt is None:
            ref_prompt = self.ref_prompt
        if ref_language is None:
            ref_language = self.ref_language

        ref_language = LANG.from_string(ref_language) if isinstance(ref_language, str) else ref_language
        if ref_language not in list(LANG):
            raise ValueError(f"ref_language must be in {list(LANG)}")
        ref_language = str(ref_language)
        text_language = LANG.from_string(text_language) if isinstance(text_language, str) else text_language
        if text_language not in list(LANG):
            raise ValueError(f"text_language must be in {list(LANG)}")
        text_language = str(text_language)

        ref_prompt = ref_prompt.strip()
        text = text.strip()
        if ref_prompt[-1] not in sentence_split_symbols:
            ref_prompt += "。" if ref_language != "en" else "."
        if text[0] not in sentence_split_symbols and len(get_first(text)) < 4:
            text = "。" + text if text_language != "en" else "." + text

        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.half else np.float32,
        )
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise ValueError("参考音频需要是3~10秒范围内的，请更换！")
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.half:
            wav16k = wav16k.half().to(self.device)
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
        wav16k = torch.cat([wav16k, zero_wav_torch])

        ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]

        # step1
        phones1, word2ph1, norm_text1 = self.get_cleaned_text_final(ref_prompt, ref_language)
        text = sentence_split_by_length(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = merge_short_text_in_array(texts, 5)

        audio_list = []
        bert1 = self.get_bert_final(phones1, word2ph1, norm_text1, ref_language).to(self.dtype)

        for text in texts:
            if not text.strip():
                continue
            if text[-1] not in sentence_split_symbols:
                text += "。" if text_language != "en" else "."
            logger.debug(f"实际输入的目标文本(每句): {text}")
            phones2, word2ph2, norm_text2 = self.get_cleaned_text_final(text, text_language)
            bert2 = self.get_bert_final(phones2, word2ph2, norm_text2, text_language).to(self.dtype)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            # step2
            pred_semantic, idx = self.t2s_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=50 * self.t2s_max_sec,
            )

            # step3
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = get_spepc(self.hps, ref_wav_path)
            refer = self.to_device(refer)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    refer,
                ).detach().cpu().numpy()[0, 0]
            )  # 重建不带上prompt部分
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio /= max_audio
            audio_list.append(audio)
        audio_output = self.audio_numpy_concat(audio_list, sr=self.sampling_rate, speed=speed)
        if output_path is None:
            return audio_output
        else:
            soundfile.write(output_path, audio_output, self.sampling_rate)
            logger.debug(f"Save audio to {output_path}")
            return output_path

    @torch.inference_mode()
    def prepare_reference(
            self,
            ref_wav_path: str = None,
            ref_prompt: str = None,
            ref_language: Union[str, LANG] = None,
    ):
        """
        Prepare and cache reference audio features for streaming TTS.
        This significantly reduces first chunk latency by pre-computing reference features.
        
        Args:
            ref_wav_path: str, path to the reference wav file
            ref_prompt: str, reference prompt text
            ref_language: str, language of the reference prompt
            
        Returns:
            cache_key: str, key for accessing cached features
        """
        if ref_wav_path is None:
            ref_wav_path = self.ref_wav_path
        if ref_prompt is None:
            ref_prompt = self.ref_prompt
        if ref_language is None:
            ref_language = self.ref_language
            
        ref_language = LANG.from_string(ref_language) if isinstance(ref_language, str) else ref_language
        ref_language = str(ref_language)
        
        ref_prompt = ref_prompt.strip()
        if ref_prompt[-1] not in sentence_split_symbols:
            ref_prompt += "。" if ref_language != "en" else "."
            
        # Create cache key
        cache_key = f"{ref_wav_path}_{ref_prompt}_{ref_language}"
        
        # Check if already cached
        if cache_key in self._ref_cache:
            logger.debug(f"Reference already cached: {cache_key}")
            return cache_key
            
        logger.info(f"Preparing reference audio features for streaming...")
        
        # Load and process reference audio
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.half else np.float32,
        )
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise ValueError("参考音频需要是3~10秒范围内的，请更换！")
            
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.half:
            wav16k = wav16k.half().to(self.device)
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        
        # Extract semantic features
        ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        
        # Process reference text
        phones1, word2ph1, norm_text1 = self.get_cleaned_text_final(ref_prompt, ref_language)
        bert1 = self.get_bert_final(phones1, word2ph1, norm_text1, ref_language).to(self.dtype)
        
        # Get reference spectrogram
        refer = get_spepc(self.hps, ref_wav_path)
        refer = self.to_device(refer)
        
        # Cache all features
        self._ref_cache[cache_key] = {
            "prompt_semantic": prompt_semantic,
            "phones1": phones1,
            "word2ph1": word2ph1,
            "norm_text1": norm_text1,
            "bert1": bert1,
            "refer": refer,
            "ref_language": ref_language,
        }
        
        logger.info(f"Reference prepared and cached: {cache_key}")
        return cache_key

    @torch.inference_mode()
    def predict_stream(
            self,
            text: str,
            text_language: Union[str, LANG] = "auto",
            speed: float = 1.0,
            ref_wav_path: str = None,
            ref_prompt: str = None,
            ref_language: Union[str, LANG] = None,
            top_k: int = 20,
            top_p: float = 0.6,
            temperature: float = 0.6,
            stream_chunk_size: int = 20,
    ):
        """
        Streaming TTS generation with incremental decoding.
        Accumulates all semantic tokens and decodes incrementally, outputting only new audio.
        
        Args:
            text: str, target text
            text_language: str, language of target text
            speed: float, speech speed
            ref_wav_path: str, reference audio file
            ref_prompt: str, reference prompt text
            ref_language: str, reference language
            top_k: int, top k sampling
            top_p: float, top p sampling
            temperature: float, sampling temperature
            stream_chunk_size: int, number of semantic tokens per chunk (smaller = lower latency)
            
        Yields:
            audio_chunk: numpy array, incremental audio chunk (float32)
        """
        # Prepare reference (use cache if available)
        cache_key = self.prepare_reference(ref_wav_path, ref_prompt, ref_language)
        ref_cache = self._ref_cache[cache_key]
        
        # Parse language
        text_language = LANG.from_string(text_language) if isinstance(text_language, str) else text_language
        text_language = str(text_language)
        
        # Preprocess text
        text = text.strip()
        if text[0] not in sentence_split_symbols and len(get_first(text)) < 4:
            text = "。" + text if text_language != "en" else "." + text
            
        text = sentence_split_by_length(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = merge_short_text_in_array(texts, 5)
        
        logger.debug(f"Streaming TTS for {len(texts)} text segments")
        
        # Process each text segment
        for text_idx, text in enumerate(texts):
            if not text.strip():
                continue
                
            if text[-1] not in sentence_split_symbols:
                text += "。" if text_language != "en" else "."
                
            logger.debug(f"Processing segment {text_idx + 1}/{len(texts)}: {text}")
            
            # Text processing
            phones2, word2ph2, norm_text2 = self.get_cleaned_text_final(text, text_language)
            bert2 = self.get_bert_final(phones2, word2ph2, norm_text2, text_language).to(self.dtype)
            bert = torch.cat([ref_cache["bert1"], bert2], 1)
            
            all_phoneme_ids = torch.LongTensor(ref_cache["phones1"] + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = ref_cache["prompt_semantic"].unsqueeze(0).to(self.device)
            
            # Streaming with chunk-by-chunk decoding
            # Strategy: decode each semantic chunk independently to avoid alignment issues
            for semantic_chunk in self._generate_semantic_stream(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                chunk_size=stream_chunk_size,
            ):
                try:
                    # Decode only the current semantic chunk
                    # This avoids alignment issues between accumulated semantics and phonemes
                    audio_chunk = self.vq_model.decode(
                        semantic_chunk.unsqueeze(0),
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                        ref_cache["refer"],
                    ).detach().cpu().numpy()[0, 0]
                    
                    # Normalize audio
                    max_audio = np.abs(audio_chunk).max()
                    if max_audio > 1:
                        audio_chunk /= max_audio
                    
                    # Apply speed adjustment if needed
                    if speed != 1.0:
                        audio_chunk = librosa.effects.time_stretch(audio_chunk, rate=speed, sr=self.sampling_rate)
                    
                    # Yield the audio chunk
                    if len(audio_chunk) > 0:  # Only yield non-empty chunks
                        yield audio_chunk.astype(np.float32)
                    
                except Exception as e:
                    logger.warning(f"Failed to decode chunk: {e}")
                    continue

    def _generate_semantic_stream(
            self,
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            top_k=20,
            top_p=0.6,
            temperature=0.6,
            chunk_size=20,
    ):
        """
        Generate semantic tokens in streaming fashion.
        Yields chunks of semantic tokens as they are generated.
        (Legacy method - kept for compatibility)
        """
        from parrots.t2s_utils import sample
        import torch.nn.functional as F
        
        # Prepare text embeddings
        x = self.t2s_model.ar_text_embedding(all_phoneme_ids)
        x = x + self.t2s_model.bert_proj(bert.transpose(1, 2))
        x = self.t2s_model.ar_text_position(x)
        
        y = prompt
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device)
        
        # Cache for efficient generation
        cache = {
            "all_stage": self.t2s_model.num_layers,
            "k": [None] * self.t2s_model.num_layers,
            "v": [None] * self.t2s_model.num_layers,
            "y_emb": None,
            "first_infer": 1,
            "stage": 0,
        }
        
        chunk_tokens = []
        
        for idx in range(1500):  # Max length
            # Generate next token
            if cache["first_infer"] == 1:
                y_emb = self.t2s_model.ar_audio_embedding(y)
            else:
                y_emb = torch.cat([cache["y_emb"], self.t2s_model.ar_audio_embedding(y[:, -1:])], 1)
            cache["y_emb"] = y_emb
            
            y_pos = self.t2s_model.ar_audio_position(y_emb)
            
            if cache["first_infer"] == 1:
                xy_pos = torch.concat([x, y_pos], dim=1)
            else:
                xy_pos = y_pos[:, -1:]
                
            y_len = y_pos.shape[1]
            
            if cache["first_infer"] == 1:
                x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
                y_attn_mask = F.pad(
                    torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=y.device), diagonal=1),
                    (x_len, 0),
                    value=False,
                )
                xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            else:
                xy_attn_mask = torch.zeros((1, x_len + y_len), dtype=torch.bool, device=xy_pos.device)
                
            xy_dec, _ = self.t2s_model.h((xy_pos, None), mask=xy_attn_mask, cache=cache)
            logits = self.t2s_model.ar_predict_layer(xy_dec[:, -1])
            
            if idx == 0:
                logits = logits[:, :-1]
                
            samples = sample(
                logits[0], y, top_k=top_k, top_p=top_p, repetition_penalty=1.35, temperature=temperature
            )[0].unsqueeze(0)
            
            # Check for EOS
            if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
                if len(chunk_tokens) > 0:
                    yield torch.cat(chunk_tokens, dim=1)
                break
                
            y = torch.concat([y, samples], dim=1)
            cache["first_infer"] = 0
            
            # Accumulate tokens for chunk
            if y.shape[1] > prefix_len:
                chunk_tokens.append(y[:, -1:])
                
                # Yield chunk when reaching chunk_size
                if len(chunk_tokens) >= chunk_size:
                    yield torch.cat(chunk_tokens, dim=1)
                    chunk_tokens = []


if __name__ == "__main__":
    sys.path.append('..')
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
        default="pretrained_models/s2G488k.pth",
        help="Path to the pretrained SoVITS",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        help="Path to the pretrained GPT",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--half", action="store_true", help="Use half precision instead of float32")
    parser.add_argument("--text", type=str, default="你好，欢迎来北京。welcome to the city.", help="input text")
    parser.add_argument("--lang", type=str, default="auto", help="Language of the text, zh, en, jp, auto")
    parser.add_argument("--ref_wav_path", type=str, default="../examples/ref.wav", help="reference wav")
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

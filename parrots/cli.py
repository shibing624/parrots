# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cli
"""
import sys
from typing import Union

import fire
import numpy as np

sys.path.append('..')

from parrots.asr import SpeechRecognition
from parrots.tts import TextToSpeech
from parrots.log import logger

global speech_recognition_model
speech_recognition_model = None


def asr_func(audio_file_path: Union[np.ndarray, bytes, str], output_path: str = None, **kwargs):
    """
    Compute ASR result from audio file.
    """
    global speech_recognition_model
    if speech_recognition_model is None:
        speech_recognition_model = SpeechRecognition(**kwargs)
    r = speech_recognition_model.predict(audio_file_path)
    res_text = r.get("text", "")
    print(res_text, end='')  # Stdout
    if output_path:
        with open(output_path, "w") as f:
            f.write(res_text)
        logger.debug(f"ASR done, result saved: {output_path}")


global text_to_speech_model
text_to_speech_model = None


def tts_func(text: str, output_path: str, text_language: str = 'auto', **kwargs):
    """
    Compute TTS result from text.
    """
    global text_to_speech_model
    if text_to_speech_model is None:
        text_to_speech_model = TextToSpeech(**kwargs)
    text_to_speech_model.predict(
        text=text,
        text_language=text_language,
        output_path=output_path
    )
    logger.debug(f"TTS done, result saved: {output_path}")


def main():
    """Main entry point"""

    fire.Fire(
        {
            "asr": asr_func,
            "tts": tts_func,
        }
    )


if __name__ == "__main__":
    main()

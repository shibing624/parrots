# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import threading
import time
import wave
from pathlib import Path

import pyaudio
import pypinyin
from loguru import logger
from pydub import AudioSegment
from pypinyin import lazy_pinyin

from parrots.num_util import num2chinese

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_syllables_dir = os.path.join(pwd_path, 'data/syllables')


class TextToSpeech(object):
    def __init__(self, syllables_dir=default_syllables_dir):
        self.syllables_dir = syllables_dir
        # TODO: 分数的读法 2.11 待修复，如何添加'.'
        self.punctuation = ['，', '。', '？', '！', '“', '”', '；', '：', '(', '）',
                            '.', ':', ';', ',', '?', '!', '\"', "\'", '(', ')']

    def update_syllables_dir(self, new_dir):
        self.syllables_dir = new_dir

    def _play_audio(self, path, delay, chunk=1024):
        try:
            time.sleep(delay)
            wf = wave.open(path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(chunk)

            while data:
                stream.write(data)
                data = wf.readframes(chunk)

            stream.stop_stream()
            stream.close()

            p.terminate()
            return
        except IOError as ioe:
            logger.warning(ioe)
        except Exception as e:
            logger.warning(e)

    def speak(self, text):
        syllables = lazy_pinyin(text, style=pypinyin.TONE3)
        logger.debug(syllables)
        delay = 0

        def preprocess(syllables):
            temp = []
            for syllable in syllables:
                for p in self.punctuation:
                    syllable = syllable.replace(p, '')
                if syllable.isdigit():
                    syllable = num2chinese(syllable)
                    new_sounds = lazy_pinyin(syllable, style=pypinyin.TONE3)
                    for e in new_sounds:
                        temp.append(e)
                else:
                    temp.append(syllable)
            return temp

        syllables = preprocess(syllables)
        threads = []
        for syllable in syllables:
            path = os.path.join(self.syllables_dir, syllable + ".wav")
            if not os.path.exists(path): continue
            t = threading.Thread(target=self._play_audio, args=(path, delay))
            threads.append(t)
            delay += 0.355
        for t in threads:
            t.start()

    def synthesize(self, input_text='', output_wav_path=''):
        """
        Synthesize .wav from text
        input_text: the folder that contains all syllables .wav files
        output_wav_path: the destination folder to save the synthesized file
        """
        delay = 0
        increment = 355  # milliseconds
        pause = 500  # pause for punctuation
        syllables = lazy_pinyin(input_text, style=pypinyin.TONE3)

        # initialize to be complete silence, each character takes up ~500ms
        result = AudioSegment.silent(duration=500 * len(input_text))
        for syllable in syllables:
            path = os.path.join(self.syllables_dir, syllable + ".wav")
            sound_file = Path(path)
            # insert 500 sr silence for punctuation marks
            if syllable in self.punctuation:
                short_silence = AudioSegment.silent(duration=pause)
                result = result.overlay(short_silence, position=delay)
                delay += increment
                continue
            # skip sound file that doesn't exist
            if not sound_file.is_file():
                continue
            segment = AudioSegment.from_wav(path)
            result = result.overlay(segment, position=delay)
            delay += increment
        if not output_wav_path:
            output_wav_path = 'out.wav'

        result.export(output_wav_path, format="wav")
        logger.debug(f"Exported: {output_wav_path}")
        return result

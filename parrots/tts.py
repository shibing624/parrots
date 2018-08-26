# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import _thread
import os
import time
import wave
from pathlib import Path

import pyaudio
import pypinyin
from pydub import AudioSegment
from pypinyin import lazy_pinyin

from parrots.num_util import num2chinese


class TextToSpeech:
    CHUNK = 1024
    punctuation = ['，', '。', '？', '！', '“', '”', '；', '：', '（', "）", ":", \
                   ";", ",", ".", "?", "!", "\"", "\'", "(", ")"]

    def __init__(self):
        pass

    def speak(self, text):
        syllables = lazy_pinyin(text, style=pypinyin.TONE3)
        print(syllables)
        delay = 0

        def preprocess(syllables):
            temp = []
            for syllable in syllables:
                for p in TextToSpeech.punctuation:
                    syllable = syllable.replace(p, "")
                if syllable.isdigit():
                    syllable = num2chinese(syllable)
                    new_sounds = lazy_pinyin(syllable, style=pypinyin.TONE3)
                    for e in new_sounds:
                        temp.append(e)
                else:
                    temp.append(syllable)
            return temp

        syllables = preprocess(syllables)
        for syllable in syllables:
            path = "syllables/" + syllable + ".wav"
            _thread.start_new_thread(TextToSpeech._play_audio, (path, delay))
            delay += 0.355

    def synthesize(self, text, src, dst):
        """
        Synthesize .wav from text
        src is the folder that contains all syllables .wav files
        dst is the destination folder to save the synthesized file
        """
        print("Synthesizing ...")
        delay = 0
        increment = 355  # milliseconds
        pause = 500  # pause for punctuation
        syllables = lazy_pinyin(text, style=pypinyin.TONE3)

        # initialize to be complete silence, each character takes up ~500ms
        result = AudioSegment.silent(duration=500 * len(text))
        for syllable in syllables:
            path = src + syllable + ".wav"
            print(path)
            sound_file = Path(path)
            # insert 500 ms silence for punctuation marks
            if syllable in TextToSpeech.punctuation:
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

        directory = dst
        if not os.path.exists(directory):
            os.makedirs(directory)

        result.export(directory + "generated.wav", format="wav")
        print("Exported.")

    def _play_audio(path, delay):
        try:
            time.sleep(delay)
            wf = wave.open(path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            data = wf.readframes(TextToSpeech.CHUNK)

            while data:
                stream.write(data)
                data = wf.readframes(TextToSpeech.CHUNK)

            stream.stop_stream()
            stream.close()

            p.terminate()
            return
        except:
            pass


if __name__ == '__main__':
    tts = TextToSpeech()

    while True:
        tts.speak(input('输入中文：'))

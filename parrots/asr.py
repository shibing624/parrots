# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

has_cuda = torch.cuda.is_available()


class SpeechRecognition:
    def __init__(
            self,
            model_name_or_path: str = "BELLE-2/Belle-distilwhisper-large-v2-zh",
            use_cuda: Optional[bool] = has_cuda,
            cuda_device: Optional[int] = -1,
            max_new_tokens: Optional[int] = 128,
            chunk_length_s: Optional[int] = 15,
            batch_size: Optional[int] = 16,
            torch_dtype: Optional[str] = "float16",
            use_flash_attention_2: Optional[bool] = False,
            language: Optional[str] = "zh",
            **kwargs
    ):
        """
        Initialize the speech recognition object.
        :param model_name_or_path: Model name or path, like:
            'BELLE-2/Belle-distilwhisper-large-v2-zh', 'distil-whisper/distil-large-v2', ...
            model in HuggingFace Model Hub and release from
        :param use_cuda: Whether or not to use CUDA for inference.
        :param cuda_device: Which cuda device to use for inference.
        :param max_new_tokens: The maximum number of new tokens to generate, ignoring the number of tokens in the
            prompt.
        :param chunk_length_s: The length in seconds of the audio chunks to feed to the model.
        :param batch_size: The batch size to use for inference.
        :param torch_dtype: The torch dtype to use for inference.
        :param use_flash_attention_2: Whether or not to use the FlashAttention2 module.
        :param language: The language of the model to use.
        :param kwargs: Additional keyword arguments passed along to the pipeline.
        """
        self.device_map = "auto"
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
                    self.device_map = {"": int(cuda_device)}
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_map = {"": "mps"}
            else:
                self.device = "cpu"
                self.device_map = {"": "cpu"}

        torch_dtype = (
            torch_dtype
            if torch_dtype in ["auto", None]
            else getattr(torch, torch_dtype)
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attention_2=use_flash_attention_2,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            chunk_length_s=chunk_length_s,
            **kwargs
        )
        if language == 'zh':
            self.pipe.model.config.forced_decoder_ids = (
                self.pipe.tokenizer.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe"
                )
            )
        logger.debug(f"Speech recognition model: {model_name_or_path} has been loaded.")

    def predict(self, inputs: Union[np.ndarray, bytes, str]):
        """语音识别用的函数，识别一个wav序列的语音
        Transcribe the audio sequence(s) given as inputs to text. See the [`AutomaticSpeechRecognitionPipeline`]
        documentation for more information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is either the filename of a local audio file, or a public URL address to download the
                      audio file. The file will be read at the correct sampling rate to get the waveform using
                      *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.
            return_timestamps (*optional*, `str` or `bool`):
                Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for
                other sequence-to-sequence models.

                For CTC models, timestamps can take one of two formats:
                    - `"char"`: the pipeline will return timestamps along the text for every character in the text. For
                        instance, if you get `[{"text": "h", "timestamp": (0.5, 0.6)}, {"text": "i", "timestamp": (0.7,
                        0.9)}]`, then it means the model predicts that the letter "h" was spoken after `0.5` and before
                        `0.6` seconds.
                    - `"word"`: the pipeline will return timestamps along the text for every word in the text. For
                        instance, if you get `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp":
                        (1.0, 1.5)}]`, then it means the model predicts that the word "hi" was spoken after `0.5` and
                        before `0.9` seconds.

                For the Whisper model, timestamps can take one of two formats:
                    - `"word"`: same as above for word-level CTC timestamps. Word-level timestamps are predicted
                        through the *dynamic-time warping (DTW)* algorithm, an approximation to word-level timestamps
                        by inspecting the cross-attention weights.
                    - `True`: the pipeline will return timestamps along the text for *segments* of words in the text.
                        For instance, if you get `[{"text": " Hi there!", "timestamp": (0.5, 1.5)}]`, then it means the
                        model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                        Note that a segment of text refers to a sequence of one or more words, rather than individual
                        words as with word-level timestamps.
            generate_kwargs (`dict`, *optional*):
                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
                complete overview of generate, check the [following
                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation).
            max_new_tokens (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

        Return:
            `Dict`: A dictionary with the following keys:
                - **text** (`str`): The recognized text.
                - **chunks** (*optional(, `List[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
                    "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.

        example:
        ```
            for sample in tqdm(dataset):
                audio = sample["audio"]
                inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
                inputs = inputs.to(device=device, dtype=torch.float16)

                outputs = model.generate(**inputs)
                print(processor.batch_decode(outputs, skip_special_tokens=True))
        ```
        """

        return self.pipe(inputs)

    def recognize_speech_from_file(self, filename):
        """
        语音识别用的接口函数
        :param filename: 识别指定文件名的语音
        :return:
        """
        return self.predict(filename)

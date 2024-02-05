# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from parrots import chinese_utils, japanese_utils, english_utils
from parrots.symbols import symbols

symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text):
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    """
    phones = [symbol_to_id[symbol] for symbol in cleaned_text]
    return phones


language_module_map = {"zh": chinese_utils, "ja": japanese_utils, "en": english_utils}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_text(text, language='zh'):
    """Clean text to sequence."""
    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


if __name__ == "__main__":
    print(len(symbols))
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))

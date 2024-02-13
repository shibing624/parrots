# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='parrots',
    version='1.0.2',
    description='Parrots, Automatic Speech Recognition(**ASR**), Text-To-Speech(**TTS**) toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/parrots',
    license="Apache 2.0",
    zip_safe=False,
    python_requires=">=3.6.0",
    entry_points={"console_scripts": ["parrots = parrots.cli:main"]},
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='TTS,ASR,text to speech,speech',
    install_requires=[
        'pypinyin',
        'jieba',
        'loguru',
        'transformers',
        'huggingface_hub',
        'librosa',
        'nltk',
        'g2p_en',
        'cn2an',
        'zh-normalization',
        'einops',
        'LangSegment',
        'soundfile',
        'fire',
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'parrots': 'parrots'},
    package_data={'parrots': ['*.*', 'data/*', 'data/pinyin2hanzi/*']}
)

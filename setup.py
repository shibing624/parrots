# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
from setuptools import setup, find_packages

readme = ''
if os.path.exists("README.md"):
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()

setup(
    name='parrots',
    version='1.2.5',
    description='Parrots, Automatic Speech Recognition(**ASR**), Text-To-Speech(**TTS**) toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/parrots',
    license="Apache 2.0",
    zip_safe=False,
    python_requires=">=3.8.0",
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
        'soundfile',
        'fire',
        'tqdm',
        'descript-audiotools',
        'torchaudio',
        'munch',
        'wetext',
        'pandas',
        'sentencepiece',
    ],
    packages=find_packages(exclude=['tests', 'examples', '*.ipynb_checkpoints', '*.__pycache__']),
    package_dir={'parrots': 'parrots'},
    package_data={
        'parrots': ['*.*', 'data/*', 'data/pinyin2hanzi/*'],
        # indextts 目录下的所有文件都打包（包括 .py, .cpp, .cu, .h, .json 等所有文件）
        'parrots.indextts': ['**/*'],
    },
    include_package_data=True,
)

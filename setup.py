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
    version='0.1.6',
    description='Chinese Text To Speech and Speech Recognition',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/parrots',
    license="Apache 2.0",
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
    install_requires=['pypinyin',
                      'jieba'],
    packages=find_packages(exclude=['tests']),
    package_dir={'parrots': 'parrots'},
    package_data={
        'parrots': ['*.*', 'README.*', 'data/*', 'utils/*', 'data/pinyin2hanzi/*', 'data/speech_model/*']}
)

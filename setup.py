# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 
from __future__ import print_function
from setuptools import setup, find_packages
from parrots import __version__

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='parrots',
    version=__version__,
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='TTS,ASR,text to speech,speech',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'parrots': 'parrots'},
    package_data={
        'parrots': ['*.*', 'LICENSE', 'README.*', 'data/*', 'utils/*', 'data/pinyin2hanzi/*', 'data/speech_model/*']}
)

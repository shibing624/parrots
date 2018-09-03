# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""
import os
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Reshape  # , Flatten
from keras.layers import Lambda, Activation, Conv2D, MaxPooling2D  # , Merge
from keras.models import Model
from keras.optimizers import Adam

from parrots.utils.file_reader import get_pinyin_list
from parrots.utils.io_util import get_logger
from parrots.utils.wav_util import get_frequency_features, read_wav_data

default_logger = get_logger(__file__)


class SpeechRecognition(object):
    def __init__(self, pinyin_path='./data/pinyin_hanzi_dict.txt',
                 model_path='./data/speech_model/speech_recognition.model'):
        """
        初始化
        :param pinyin_path: pinyin file path
        """
        MS_OUTPUT_SIZE = 1422  # 默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.create_model()
        self.initialized = False
        self.pinyin_path = pinyin_path
        self.model_path = model_path

    def initialize(self):
        pwd_path = os.path.abspath(os.path.dirname(__file__))
        if self.pinyin_path:
            t1 = time.time()
            try:
                self.pinyin_list = get_pinyin_list(self.pinyin_path)  # 获取拼音列表
            except IOError:
                pinyin_path = os.path.join(pwd_path, '..', self.pinyin_path)
                self.pinyin_list = get_pinyin_list(pinyin_path)  # 获取拼音列表
            default_logger.debug(
                "Loading pinyin dict cost %.3f seconds." % (time.time() - t1))
        if self.model_path:
            t2 = time.time()
            try:
                self._model.load_weights(self.model_path)
                self.base_model.load_weights(self.model_path + '.base')
            except IOError:
                model_path = os.path.join(pwd_path, '..', self.model_path)
                self._model.load_weights(model_path)
                self.base_model.load_weights(model_path + '.base')
            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t2))
            default_logger.debug("Speech recognition model has been built ok.")
            self.graph = tf.get_default_graph()

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def create_model(self):
        """
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        :return:
        """
        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        # layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
        layer_h3 = Dropout(0.05)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        layer_h6 = Dropout(0.1)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h9)
        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h12)
        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        layer_h16 = Reshape((200, 3200))(layer_h15)  # Reshape层
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        # model.summary()

        # clip norm seems to speeds up convergence
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        default_logger.debug('Create Model Successful, Compiles Model Successful. ')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def predict(self, data_input, input_len):
        """
        预测结果
        :param data_input:
        :param input_len:
        :return: 返回语音识别后的拼音符号列表
        """
        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)
        in_len[0] = input_len
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input
        with self.graph.as_default():
            base_pred = self.base_model.predict(x=x_in)
        base_pred = base_pred[:, :, :]
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)
        r1 = K.get_value(r[0][0])
        # print('r1', r1)
        r1 = r1[0]
        return r1

    def recognize_speech(self, wavsignal, fs):
        """
        语音识别用的函数，识别一个wav序列的语音
        :param wavsignal:
        :param fs:
        :return:
        """
        self.check_initialized()
        result = []
        data_input = get_frequency_features(wavsignal, fs)
        input_length = len(data_input)
        input_length = input_length // 8
        data_input = np.array(data_input, dtype=np.float)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        preds = self.predict(data_input, input_length)
        for i in preds:
            result.append(self.pinyin_list[i])
        return result

    def recognize_speech_from_file(self, filename):
        """
        语音识别用的接口函数
        :param filename: 识别指定文件名的语音
        :return:
        """
        signal, fs = read_wav_data(filename)
        return self.recognize_speech(signal, fs)

    @property
    def model(self):
        """
        model
        :return: keras model
        """
        self.check_initialized()
        return self._model

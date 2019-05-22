# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cnn_model_2input
   Description :
   Author :       xszhou
   date：          2019/2/19
-------------------------------------------------
"""
__author__ = 'xszhou'

# 将原始输入改用双输入形式，字符级和词语级双输入

import tensorflow as tf
import numpy as np
import data.data_loader
import data.data_loader_wordlevel


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度，这里字和词的向量都定为100维
    seq_length_c = 600  # 字符级序列长度
    seq_length_w = 600  # 词语级序列长度
    num_classes = 2  # 类别数
    num_filters = 64  # 卷积核数目
    # kernel_sizes = "3,4,5,6,7"  # 卷积核尺寸
    kernel_sizes = "5,6,7"  # 卷积核尺寸
    # kernel_sizes_w = "2,3,4"  # 词语级卷积核尺寸
    kernel_sizes_w = "3,4,5"  # 词语级卷积核尺寸
    vocab_size_c = 100000  # 字表大小
    vocab_size_w = 100000  # 词表大小
    hidden_dim = 128  # 全连接层神经元
    hidden_size = 128  # 隐态维度，应当是卷积核种类数*每种种类的数量
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 20  # 每多少个batch输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    num_layers_for_rnn = 2  # rnn的层数，这里设置成2

    rnn = 'gru'

    weight = []

    attention_size = 1


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x_c = tf.placeholder(tf.int32, [None, self.config.seq_length_c], name='input_x_c')
        self.input_x_w = tf.placeholder(tf.int32, [None, self.config.seq_length_w], name='input_x_w')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()



    def cnn(self):
        """CNN模型"""

        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)

        # 词向量映射
        # 这里暂时用随机初始化的词向量
        # print(self.config.weight)
        embedding_w = np.array(self.config.weight, dtype='float32')
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            embedding_c = tf.get_variable('embedding_c', [self.config.vocab_size_c, self.config.embedding_dim])
            # embedding_w = tf.get_variable('embedding_w', [self.config.vocab_size_w, self.config.embedding_dim])

            # embedding = tf.convert_to_tensor(embedding_w)
            embedding_w = tf.get_variable("embedding_update", shape=[self.config.vocab_size_w, self.config.embedding_dim],
                                        initializer=tf.constant_initializer(embedding_w), trainable=True)
            # embedding_notrain = tf.get_variable("embedding_weight", embedding, trainable=False)
            self.embedding_inputs_c = tf.nn.embedding_lookup(embedding_c, self.input_x_c)
            self.embedding_inputs_w = tf.nn.embedding_lookup(embedding_w, self.input_x_w)


        filters_w = str(self.config.kernel_sizes_w).split(',')
        # 多类型卷积核-词语级
        pooled_outputs_w = []
        max_feature_length = self.config.seq_length_w - 5 + 1
        for filter_size in filters_w:
            # each conv-pool
            scope_name = str('conv-maxpool-%s-w' % filter_size)
            with tf.device('/gpu:0'), tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs_w, filters=self.config.num_filters,
                                        kernel_size=int(filter_size), name='conv%s-w' % filter_size)
                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                h_reshape = conv[:, :max_feature_length, :]
            pooled_outputs_w.append(h_reshape)



        # combine
        filters_pattern = str(self.config.kernel_sizes_w).split(',')
        if len(filters_pattern) > 1:
            rnn_input = tf.concat(pooled_outputs_w, -1)
        else:
            rnn_input = conv

        with tf.device('/gpu:0'), tf.name_scope("rnn"):
            cells_fw = [dropout() for _ in range(self.config.num_layers_for_rnn)]
            cells_bw = [dropout() for _ in range(self.config.num_layers_for_rnn)]
            rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
            rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
            _outputs, _state = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, inputs=rnn_input,
                                                               dtype=tf.float32)
            self.final_state = _state
            self.final_output = tf.concat(_outputs, 2)
        # *************************************************************************************************************

        with tf.device('/gpu:0'), tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            last = tf.reduce_mean(self.final_output, 1)  # 取所有时刻输出的平均
            # fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # fc = tf.nn.relu(fc)
            # fc = tf.nn.tanh(fc)

            # 分类器
            self.logits = tf.layers.dense(last, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

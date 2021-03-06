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
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 20  # 每多少个batch输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

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



        filters = str(self.config.kernel_sizes).split(',')
        # 多类型卷积核-字符级
        pooled_outputs_c = []
        for filter_size in filters:
            # each conv-pool
            scope_name = str('conv-maxpool-%s-c' % filter_size)
            with tf.device('/gpu:0'), tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs_c, filters=self.config.num_filters,
                                        kernel_size=int(filter_size), name='conv%s-c' % filter_size)

                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                # conv = tf.nn.tanh(conv)
                # (?,1498,64)

                # k这里取2
                k = 2
                conv_trans = tf.transpose(conv, perm=[0, 2, 1])
                gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
                gmp = tf.transpose(gmp_trans, perm=[0, 2, 1])

                # top-k-mean池化
                gmp = tf.reduce_mean(gmp, 1)

                # 最大池化
                # gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp%s' % filter_size)


                # conv = tf.transpose(conv, perm=[0, 2, 1])
                # k = 800
                # conv, index_gmp = tf.nn.top_k(conv, k)

            # attention_scope_name = str('attention-pooling-%s-c' % filter_size)
            # # attention-pooling
            # with tf.device('/gpu:0'), tf.name_scope(attention_scope_name):
            #     seq_filter_length = conv.shape[2]
            #     conv = tf.reshape(conv, [-1, seq_filter_length])
            #     conv = tf.expand_dims(conv, axis=-1)
            #     # *******************************************************
            #     # 这里要改
            #     conv = tf.transpose(conv, [1, 0, 2])
            #     # *******************************************************
            #     att_input = tf.reshape(conv, [-1, 1])
            #     att_input_list = tf.split(att_input, seq_filter_length, 0)
            #
            #     attention_w = tf.Variable(tf.truncated_normal([1, self.config.attention_size], stddev=0.1), name='attention_w')
            #     attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            #     u_list = []
            #
            #     for t in range(seq_filter_length):
            #         u_t = tf.tanh(tf.matmul(att_input_list[t], attention_w) + attention_b)
            #         u_list.append(u_t)
            #     u_w = tf.Variable(tf.truncated_normal([self.config.attention_size, 1], stddev=0.1), name='attention_uw')
            #     attn_z = []
            #     for t in range(seq_filter_length):
            #         z_t = tf.matmul(u_list[t], u_w)
            #         attn_z.append(z_t)
            #     attn_zconcat = tf.concat(attn_z, axis=1)
            #     self.alpha = tf.nn.softmax(attn_zconcat)
            #     alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [seq_filter_length, -1, 1])
            #     final_conv = tf.reduce_sum(att_input_list * alpha_trans, 0)
            #     final_conv = tf.reshape(final_conv, [-1, self.config.num_filters, 1])
            #     # print(final_conv.shape)
            #     final_conv = tf.reshape(final_conv, [-1, self.config.num_filters])
            #     # print(final_conv.shape)

            pooled_outputs_c.append(gmp)
            # pooled_outputs_c.append(final_conv)

        filters_w = str(self.config.kernel_sizes_w).split(',')
        # 多类型卷积核-词语级
        pooled_outputs_w = []
        for filter_size in filters_w:
            # each conv-pool
            scope_name = str('conv-maxpool-%s-w' % filter_size)
            with tf.device('/gpu:0'), tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs_w, filters=self.config.num_filters,
                                        kernel_size=int(filter_size), name='conv%s-w' % filter_size)
                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                # conv = tf.nn.tanh(conv)
                # gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp%s' % filter_size)

                # k这里取2
                k = 2
                # k = 500
                # conv_trans = tf.transpose(conv, perm=[0, 2, 1])
                gmp_trans = tf.transpose(conv, perm=[0, 2, 1])
                gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
                gmp = tf.transpose(gmp_trans, perm=[0, 2, 1])

                # top-k-mean池化
                gmp = tf.reduce_mean(gmp, 1)

                # conv = tf.transpose(conv, perm=[0, 2, 1])
                # conv = gmp_trans


            # # 这里不对，如果64个filter，应该有64个attention层
            # attention_scope_name = str('attention-pooling-%s-w' % filter_size)
            # # attention-pooling
            # with tf.device('/gpu:0'), tf.name_scope(attention_scope_name):
            #     seq_filter_length = conv.shape[2]
            #     conv = tf.reshape(conv, [-1, seq_filter_length])
            #     conv = tf.expand_dims(conv, axis=-1)
            #     # *******************************************************
            #     # 这里要改
            #     conv = tf.transpose(conv, [1, 0, 2])
            #     # *******************************************************
            #
            #     att_input = tf.reshape(conv, [-1, 1])
            #     att_input_list = tf.split(att_input, seq_filter_length, 0)
            #
            #     attention_w = tf.Variable(tf.truncated_normal([1, self.config.attention_size], stddev=0.1), name='attention_w')
            #     attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            #     u_list = []
            #
            #     for t in range(seq_filter_length):
            #         u_t = tf.tanh(tf.matmul(att_input_list[t], attention_w) + attention_b)
            #         u_list.append(u_t)
            #     u_w = tf.Variable(tf.truncated_normal([self.config.attention_size, 1], stddev=0.1), name='attention_uw')
            #     attn_z = []
            #     for t in range(seq_filter_length):
            #         z_t = tf.matmul(u_list[t], u_w)
            #         attn_z.append(z_t)
            #     attn_zconcat = tf.concat(attn_z, axis=1)
            #     self.alpha = tf.nn.softmax(attn_zconcat)
            #     alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [seq_filter_length, -1, 1])
            #     final_conv = tf.reduce_sum(att_input_list * alpha_trans, 0)
            #     final_conv = tf.reshape(final_conv, [-1, self.config.num_filters, 1])
            #     # print(final_conv.shape)
            #     final_conv = tf.reshape(final_conv, [-1, self.config.num_filters])
            #     # print(final_conv.shape)

            # pooled_outputs_w.append(final_conv)
            pooled_outputs_w.append(gmp)


        # combine
        # num_filters_total = self.config.num_filters * len(filters)
        # (?, 320)
        self.pool_c = tf.concat(pooled_outputs_c, 1)
        # (?, 320)
        self.pool_w = tf.concat(pooled_outputs_w, 1)
        # no need to flat

        pooled_outputs_cw = []
        pooled_outputs_cw.append(self.pool_c)
        pooled_outputs_cw.append(self.pool_w)

        # 字词结合
        self.pool_combine_cw = tf.concat(pooled_outputs_cw, 1)

        with tf.device('/gpu:0'), tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.pool_combine_cw, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # fc = tf.nn.tanh(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
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

    # def attention(self, x_i, x, index):
    #     e_i = []
    #     c_i = []
    #     for output in x:
    #         output = tf.reshape(output, [-1, self.config.embedding_dim])
    #         atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
    #         e_i_j = tf.matmul(atten_hidden, self.attention_V)
    #         e_i.append(e_i_j)
    #     e_i = tf.concat(e_i, axis=1)
    #     # e_i = tf.exp(e_i)
    #     alpha_i = tf.nn.softmax(e_i)
    #     alpha_i = tf.split(alpha_i, self.config.seq_length_c, 1)
    #
    #     # i!=j
    #     for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
    #         if j == index:
    #             continue
    #         else:
    #             output = tf.reshape(output, [-1, self.config.embedding_dim])
    #             c_i_j = tf.multiply(alpha_i_j, output)
    #             c_i.append(c_i_j)
    #     c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.config.seq_length_c - 1, self.config.embedding_dim])
    #     c_i = tf.reduce_sum(c_i, 1)
    #     return c_i

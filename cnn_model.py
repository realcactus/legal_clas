# coding: utf-8

import tensorflow as tf
import numpy as np
from data.data_loader_wordlevel import read_vocab,read_category,batch_iter,process_file,build_vocab, get_maxlength, open_file
from data.word2vec_helper import embedding_sentences



class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 14  # 类别数
    # num_filters = 100  # 卷积核数目
    num_filters = 64  # 卷积核数目
    # num_filters = 256  # 卷积核数目
    # kernel_sizes = "5"  # 卷积核尺寸
    # kernel_sizes = "3,4,5"  # 卷积核尺寸  因为是词语级别所以尺寸适当小一点
    kernel_sizes = "3,4,5,6,7"  # 卷积核尺寸
    # vocab_size = 8000  # 词汇表达小
    vocab_size = 100000  # 词汇表达小

    hidden_dim_0 = 64
    hidden_dim = 128  # 全连接层神经元


    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 30  # 总迭代轮次

    print_per_batch = 50  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    weight = []

class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        # 试一下非等长输入？
        # self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

        self.cnn()

    def onehot_dic_build(self):
        # embedding层的onehot编码
        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        # 对于字母表中不存在的或者空的字符用全0向量代替
        embedding_dic["<PAD>"] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic


    def cnn(self):
        """CNN模型"""
        # 词向量映射
        # with open('word2vec/character/out_weight.txt', 'w', encoding='utf-8') as f:
        #     for i in self.config.weight:
        #         f.write(str(i))
        #         f.write('\n')
        # with open('word2vec/out_weight.txt', 'w', encoding='utf-8') as f:
        # with open('word2vec/fasttext_vec/out_weight.txt', 'w', encoding='utf-8') as f:
        #     for i in self.config.weight:
        #         f.write(str(i))
        #         f.write('\n')
        # print(self.config.weight)

        embedding_w = np.array(self.config.weight, dtype='float32')
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            # embedding_array, _ = self.onehot_dic_build()
            # embedding = tf.convert_to_tensor(embedding_w)
            # embedding = tf.get_variable("embedding_update", shape=[self.config.vocab_size, self.config.embedding_dim],
            #                             initializer=tf.constant_initializer(embedding_w), trainable=False)
            # embedding_notrain = tf.get_variable("embedding_weight", embedding, trainable=False)

            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        # with tf.name_scope("embedding-update"):
        #     self.embedding_update = tf.layers.dense(self.embedding_inputs, self.config.embedding_dim, name='embedding_updating')

        # 多类型卷积核
        pooled_outputs = []
        filters = str(self.config.kernel_sizes).split(',')
        for filter_size in filters:
            # each conv-pool
            scope_name = str('conv-maxpool-%s' % filter_size)
            with tf.device('/gpu:0'),tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs, filters=self.config.num_filters,
                                        kernel_size=int(filter_size), name='conv%s' % filter_size)
                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                # all-leaky ReLU
                # conv = tf.maximum(0.01 * conv, conv)

                # max pooling
                # (?, 100)
                # ******************************************************************************************
                # 最大池化
                # gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp%s' % filter_size)
                # ******************************************************************************************

                # ******************************************************************************************
                # k这里取3
                k = 2
                conv_trans = tf.transpose(conv,perm=[0,2,1])
                gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
                gmp = tf.transpose(gmp_trans,perm=[0,2,1])

                # top-k-mean池化
                gmp = tf.reduce_mean(gmp, 1)

                # top-k池化
                # shape_to_flat = gmp.get_shape().as_list()
                # gmp = tf.reshape(gmp, [-1, shape_to_flat[2] * k])
                # ******************************************************************************************

                # gmp, index_gmp = tf.nn.top_k(conv, 5)
                # shape_beforemean = gmp.get_shape().as_list()
                # print(shape_beforemean)
                # gmp = tf.reduce_mean(gmp, 2)
                # shape_aftermin = gmp.get_shape().as_list()
                # print(shape_aftermin)
                # shape_to_flat = tf.shape(gmp)

            pooled_outputs.append(gmp)

        # combine
        num_filters_total = self.config.num_filters * len(filters)
        self.pool = tf.concat(pooled_outputs, 1)
        # no need to flat
        

        with tf.device('/gpu:0'),tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.pool, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

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
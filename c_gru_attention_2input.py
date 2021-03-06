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
# RNN 但此处采用词语级别的输入
# 加attention

import tensorflow as tf
import numpy as np
import data.data_loader
import data.data_loader_wordlevel


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度，这里字和词的向量都定为100维
    seq_length_c = 600  # 字符级序列长度
    seq_length_w = 600  # 词语级序列长度
    num_classes = 14  # 类别数
    num_filters = 64  # 卷积核数目
    num_filters_w = 128 # 词语卷积核数目 因为在attention机制中，我们如果要保留最大100个值进行attention,那最好不要拼接不同的卷积核
    # kernel_sizes = "3,4,5,6,7"  # 卷积核尺寸
    kernel_sizes = "3,4,5"  # 卷积核尺寸
    # kernel_sizes_w = "2,3,4"  # 词语级卷积核尺寸
    kernel_sizes_w = "5"  # 词语级卷积核尺寸
    vocab_size_c = 100000  # 字表大小
    vocab_size_w = 100000  # 词表大小
    hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 32  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 20  # 每多少个batch输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    weight = []

    num_layers_for_rnn = 2  # rnn的层数，这里设置成2
    attention_size = 128
    rnn = 'gru'


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

        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)

        def attention_function(atten_inputs, atten_size):
            ## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
            print('attention inputs: ' + str(atten_inputs))
            max_time = int(atten_inputs.shape[1])
            print("max time length: " + str(max_time))
            combined_hidden_size = int(atten_inputs.shape[2])
            print("combined hidden size: " + str(combined_hidden_size))
            W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
            b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
            u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))

            v = tf.tanh(
                tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
            print("v: " + str(v))
            # u_omega is the summarizing question vector
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
            print("vu: " + str(vu))
            exps = tf.reshape(tf.exp(vu), [-1, max_time])
            print("exps: " + str(exps))
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            print("alphas: " + str(alphas))
            atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
            print("atten outs: " + str(atten_outs))
            return atten_outs, alphas

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
            scope_name = str('conv-maxpool-%s-c' % filter_size)
            with tf.device('/gpu:0'), tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs_c, filters=self.config.num_filters,
                                        kernel_size=int(filter_size), name='conv%s-c' % filter_size)

                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                # k这里取5
                k = 3
                conv_trans = tf.transpose(conv, perm=[0, 2, 1])
                gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
                gmp = tf.transpose(gmp_trans, perm=[0, 2, 1])

                # 摊平
                # gmp = tf.layers.flatten(gmp_trans)

                # top-k-mean池化
                gmp = tf.reduce_mean(gmp, 1)

            pooled_outputs_c.append(gmp)

        filters_w = str(self.config.kernel_sizes_w).split(',')
        # 多类型卷积核-词语级
        # max_feature_seq_length_w = self.config.seq_length_w - 5 + 1
        pooled_outputs_w = []
        for filter_size in filters_w:
            scope_name = str('conv-maxpool-%s-w' % filter_size)
            with tf.device('/gpu:0'), tf.name_scope(scope_name):
                # convolution i
                conv = tf.layers.conv1d(inputs=self.embedding_inputs_w, filters=self.config.num_filters_w,
                                        kernel_size=int(filter_size), name='conv%s-w' % filter_size)
                # 新加激活层，all-ReLU
                conv = tf.nn.relu(conv)
                # 这里先做一个k-max筛选
                # k = 300
                # conv_trans = tf.transpose(conv, perm=[0, 2, 1])
                # gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
                # h_reshape = tf.transpose(gmp_trans, perm=[0, 2, 1])

                # h_reshape = conv[:,:max_feature_seq_length_w,:]

            # pooled_outputs_w.append(h_reshape)
            pooled_outputs_w.append(conv)


        # combine
        # num_filters_total = self.config.num_filters * len(filters)
        # (?, 320)
        # self.pool_w = tf.concat(pooled_outputs_w, -1)
        if len(pooled_outputs_w) > 1:
            self.pool_w = tf.concat(pooled_outputs_w, -1)
        else:
            self.pool_w = pooled_outputs_w[0]

        with tf.device('/gpu:0'),tf.name_scope("rnn"):
            cells_fw = [dropout() for _ in range(self.config.num_layers_for_rnn)]
            cells_bw = [dropout() for _ in range(self.config.num_layers_for_rnn)]
            rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
            rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
            _outputs, _state = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,rnn_cell_bw, inputs=self.pool_w, dtype=tf.float32)
            self.state = _state
            self.output = tf.concat(_outputs,2)



        # rnn_embed_size = self.config.num_filters * len(str(self.config.kernel_sizes_w).split(','))
        # output_trans = tf.transpose(self.output, perm=[1, 0, 2])
        # seq_length = output_trans.get_shape().as_list()[0]
        # rnn_embed_size = output_trans.get_shape().as_list()[2]
        # output_trans = tf.reshape(output_trans,[-1, rnn_embed_size])
        # output_trans = tf.split(output_trans, seq_length, 0)

        # self.output = output_trans

        # 定义attention layer
        with tf.device('/gpu:0'), tf.name_scope("attention"):
            outs, alphas = attention_function(self.output, self.config.attention_size)

            # attention_w = tf.Variable(tf.truncated_normal([rnn_embed_size, self.config.attention_size], stddev=0.1),
            #                             name='attention_w')
            # attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            # u_list = []
            # for t in range(seq_length):
            #     u_t = tf.tanh(tf.matmul(self.output[t], attention_w) + attention_b)
            #     u_list.append(u_t)
            # u_w = tf.Variable(tf.truncated_normal([self.config.attention_size, 1], stddev=0.1), name='attention_uw')
            # attn_z = []
            # for t in range(seq_length):
            #     z_t = tf.matmul(u_list[t], u_w)
            #     attn_z.append(z_t)
            # # transform to batch_size * sequence_length
            # attn_zconcat = tf.concat(attn_z, axis=1)
            # self.alpha = tf.nn.softmax(attn_zconcat)
            # # transform to sequence_length * batch_size * 1 , same rank as outputs
            # alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [seq_length, -1, 1])
            # self.final_output = tf.reduce_sum(self.output * alpha_trans, 0)

            self.final_output = outs
            self.alphas = alphas

        # 字词结合
        self.pool_c_combine = tf.concat(pooled_outputs_c, 1)
        # (?, 320)
        self.pool_w_combine = self.final_output
        # no need to flat

        pooled_outputs_cw = []
        pooled_outputs_cw.append(self.pool_c_combine)
        pooled_outputs_cw.append(self.pool_w_combine)

        # 字词结合
        self.pool_combine_cw = tf.concat(pooled_outputs_cw, 1)

        with tf.device('/gpu:0'), tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # last = tf.reduce_mean(self.output, 1)  # 取所有时刻输出的平均

            fc = tf.layers.dense(self.pool_combine_cw, self.config.hidden_dim, name='fc1')
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


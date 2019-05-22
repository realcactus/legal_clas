# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_cnn_cw
   Description :
   Author :       xszhou
   date：          2019/2/19
-------------------------------------------------
"""
__author__ = 'xszhou'

# 将输入文本分成两个输入，一部分是字符级输入，一部分词语级输入

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics
from cnn_model_2input import TCNNConfig, TextCNN
# from cnn_model_2input_km import TCNNConfig, TextCNN
# 这里尝试一下word级的，到时候改回来
from data import data_loader
from data import data_loader_wordlevel
from gensim.models import KeyedVectors
from data.word2vec_helper import embedding_sentences, get_word2vec_dict

base_dir = 'data/10-fold-original-data'
train_dir_c = os.path.join(base_dir, 'data-convert/train_c_1.txt')
train_dir_w = os.path.join(base_dir, 'data-convert/train_w_1.txt')
train_dir_y = os.path.join(base_dir, 'data-convert/train_y_1.txt')
test_dir_c = os.path.join(base_dir, 'data-convert/test_c_1.txt')
test_dir_w = os.path.join(base_dir, 'data-convert/test_w_1.txt')
test_dir_y = os.path.join(base_dir, 'data-convert/test_y_1.txt')

vocab_dir_c = os.path.join(base_dir, 'vocab-c.txt')
vocab_dir_w = os.path.join(base_dir, 'vocab-w.txt')


save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch_c, x_batch_w, y_batch, keep_prob):
    feed_dict = {
        model.input_x_c: x_batch_c,
        model.input_x_w: x_batch_w,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def batch_iter_cw(x_c, x_w, y, batch_size=64):
    """按照字级和词级生成批次数据"""
    data_len = len(x_c)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle_c = x_c[indices]
    x_shuffle_w = x_w[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle_c[start_id:end_id], x_shuffle_w[start_id:end_id], y_shuffle[start_id:end_id]


def evaluate(sess, x_c_, x_w_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_c_)
    # batch_eval = data_loader.batch_iter(x_, y_, 128)
    batch_eval = batch_iter_cw(x_c_, x_w_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_c_batch, x_w_batch, y_batch in batch_eval:
        batch_len = len(x_c_batch)
        feed_dict = feed_data(x_c_batch, x_w_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def read_x_from_id(filename):
    content_list = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            content = [int(x) for x in line.strip().split()]
            content_list.append(content)
    return content_list


def read_y_from_id(filename):
    content_list = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            content = [float(x) for x in line.strip().split()]
            content_list.append(content)
    return content_list


# 暂时用不到，这里词级别采用随机初始化向量
def w2v_weight_build(words):
    embedding_w = []
    # 对于字母表中不存在的或者空的字符用全0向量代替
    embedding_w.append(np.zeros(100, dtype='float32'))
    w2vModel = KeyedVectors.load_word2vec_format('word2vec/word2vec-cbow/word2vec.txt')
    index = -1
    cnt_right = 0
    cnt_wrong = 0
    for word in words:
        index += 1
        if index == 0:
            continue
        if word not in w2vModel.wv.vocab:
            cnt_wrong += 1
            embedding_w.append(w2vModel['<UNK>'])
        else:
            cnt_right += 1
            embedding_w.append(w2vModel[word])
    embedding_w = np.array(embedding_w, dtype='float32')
    return embedding_w


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    print('#####################################################')
    print(base_dir)
    print('#####################################################')
    x_train_c = np.array(read_x_from_id(train_dir_c))
    x_train_w = np.array(read_x_from_id(train_dir_w))
    y_train = np.array(read_y_from_id(train_dir_y))

    x_val_c = np.array(read_x_from_id(test_dir_c))
    x_val_w = np.array(read_x_from_id(test_dir_w))
    y_val = np.array(read_y_from_id(test_dir_y))

    x_test_c = np.array(read_x_from_id(test_dir_c))
    x_test_w = np.array(read_x_from_id(test_dir_w))
    y_test = np.array(read_y_from_id(test_dir_y))

    # create_corpus(train_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过5000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter_cw(x_train_c, x_train_w, y_train, config.batch_size)
        for x_batch_c, x_batch_w, y_batch in batch_train:
            feed_dict = feed_data(x_batch_c, x_batch_w, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val_c, x_val_w, y_val)
                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()

    x_test_c = np.array(read_x_from_id(test_dir_c))
    x_test_w = np.array(read_x_from_id(test_dir_w))
    y_test = np.array(read_y_from_id(test_dir_y))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test_c, x_test_w, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test_c)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test_c), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x_c: x_test_c[start_id:end_id],
            model.input_x_w: x_test_w[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()

    categories, cat_to_id = data_loader.read_category()
    characters, character_to_id = data_loader.read_vocab(vocab_dir_c)
    words, word_to_id = data_loader_wordlevel.read_vocab(vocab_dir_w)
    config.vocab_size_c = len(characters)
    config.vocab_size_w = len(words)
    # config.seq_length_c = 3000
    # config.seq_length_w = 1700
    config.seq_length_c = 1500
    config.seq_length_w = 800

    # ******************************************************
    # 这块代码负责给词向量初始权重赋值
    weight = w2v_weight_build(words)
    config.weight = weight
    # ******************************************************
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()

# -*- coding: utf-8 -*-

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
from data.word2vec_helper import embedding_sentences
import re
# import io


# 因为在有个老师的服务器上只能用py2，py2的编码问题真的很让人头疼，索性写一个双版本的

# if sys.version_info[0] > 2:
#     is_py3 = True
# else:
#     reload(sys)
#     sys.setdefaultencoding("utf-8")
#     is_py3 = False

# 直接指定
is_py3 = True
# 用来在py2里面写入的时候指定编码
def native_word(word, encoding='utf-8'):
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

# 用来在py2中读入的时候指定编码
def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    # print("#######################################")
                    # print(content)
                    # print("#######################################")
                    content = str(content).strip().lower()
                    # content = re.sub(r"[0-9A-Za-z\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：． :／\-［］％。？、~@#￥%……&*（）]+", "",
                    #                  content)
                    content = list(content)
                    while ' ' in content:
                        content.remove(' ')
                    while '' in content:
                        content.remove('')
                    # remove_characters = ['0','1','2','3','4','5','6','7','8','9','，','。','、','：','．',')','(','／','％','…','［', '］']

                    # 决定把预处理工作抽出来，放在data_group
                    remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                                         '(', '／', '％', '…', '［', '］',
                                         '$', '*', '【', '】', '？', '”', '，', '。', '”', '）', '（', '；', '《', '》', '-']
                    for c in remove_characters:
                        while c in content:
                            content.remove(c)
                    contents.append(content)
                    labels.append(native_content(label))
            except:
                pass
        # print("#######################################")
        # for line in contents:
        #     for c in line:
        #         print(c)
        # print("######s#################################")
    return contents, labels


def get_maxlength(filename):
    max_value = 0
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    content = str(content).strip()
                    # content = re.sub(r"[0-9A-Za-z\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：． :／\-［］％。？、~@#￥%……&*（）]+", "",
                    #                  content)
                    # temp_list = list(str(content).lower())
                    content = list(content)
                    while ' ' in content:
                        content.remove(' ')
                    remove_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '。', '、', '：', '．', ')',
                                         '(', '／', '％', '…', '［', '］',
                                         '$', '*', '【', '】', '？', '”', '，', '。', '”', '）', '（', '；', '《', '》', '-']
                    for c in remove_characters:
                        while c in content:
                            content.remove(c)
                    max_value = max(max_value, len(content))
            except:
                pass
    return max_value


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    # 这个就是list形式的sentence
    # character2vec_dict = embedding_sentences(data_train, file_to_save='word2vec/character/character2vec.model')
    # out_file = open('word2vec/character/out_key_value.txt', 'w', encoding='utf-8')
    # for k, v in character2vec_dict.items():
    #     out_file.write(k)
    #     out_file.write(str(v) + '\n')
    # out_file.close()

    all_data = []
    for content in data_train:
        # print("#######################################")
        # for c in content:
        #     print(c)
        all_data.extend(content)

    counter = Counter(all_data)
    print(counter)
    count_pairs = counter.most_common(vocab_size - 1)
    # count_pairs = counter.most_common(3460)
    print(count_pairs)
    words, _ = list(zip(*count_pairs))
    # print("#######################################")
    # for c in words:
    #     print(c)
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<UNK>'] + list(words)
    words = ['<PAD>'] + list(words)
    words_encoding = []
    for c in words:
        words_encoding.append(native_word(c, 'utf-8'))
    # open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    open_file(vocab_dir, mode='w').write('\n'.join(words_encoding) + '\n')



def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # print('#################################')
        # fp.readline()
        # print('#################################')
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录"""
    categories = ['机械制造行业', '五金建材行业', '农林牧渔行业',
                  '化工行业', '电子通讯行业', '文体生活用品行业',
                  '农副食品行业', '纺织服饰行业', '家电行业', '其他行业',
                  '食品药品行业', '交通运输行业', '酒水饮料茶奶行业']

    # categories = ['财经', '彩票', '房产',
    #               '股票', '家居', '教育',
    #               '科技', '社会', '时尚', '时政',
    #               '体育', '星座', '游戏', '娱乐']

    # categories = ['pos', 'neg']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    # print("##############################")
    # for x in labels:
    #     print(x)

    data_id, label_id = [], []
    for i in range(len(contents)):
        # 这里我认为空格对于英文句子语义来说同样是很重要的
        temp = []
        for x in contents[i]:
            if x in word_to_id:
                temp.append(word_to_id[x])
            else:
                temp.append(word_to_id['<UNK>'])
        data_id.append(temp)
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # print(data_id[0])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        # 最后一组不到batch_size的组丢掉，因为rnn好像不允许这样
        # if (i + 1) * batch_size <= data_len:
        #     end_id = (i + 1) * batch_size
        #     yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]




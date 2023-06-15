# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : create_batch_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-08 09:29:19
"""

import random
import os
import glob
import numpy as np
from sklearn import preprocessing
from utils import create_word2vec, files_processing


def get_data_batch(file_list,labels_nums,batch_size=None, shuffle=False,one_hot=False):
    '''
    加载*.npy文件的数据，循环产生批量数据batch，其中*.npy文件的数据保存是二维矩阵Mat:
    二维矩阵Mat:第一列为样本的labels,剩余的列为样本的数据，
    np.concatenate([label,data], axis=1)
    :param file_list: *.npy文件路径,type->list->[file0.npy,file1.npy,....]
    :param labels_nums: labels种类数
    :param batch_size: batch大小
    :param shuffle: 是否打乱数据,PS:只能打乱一个batch的数据，不同batch的数据不会干扰
    :param one_hot: 是否独热编码
    :return: 返回一个batch数据
    '''
    height = 0
    indexMat_labels = None
    i = 0
    while True:
        while height < batch_size:
            i = i%len(file_list)
            tempFile = file_list[i]
            tempMat_labels = np.load(tempFile)
            if indexMat_labels is None:
                indexMat_labels = tempMat_labels
            else:
                indexMat_labels = np.concatenate([indexMat_labels, tempMat_labels], 0)
            i=i+1
            height = indexMat_labels.shape[0]

        indices = list(range(height))
        batch_indices = np.asarray(indices[0:batch_size])  # 产生一个batch的index
        if shuffle:
            random.seed(100)
            random.shuffle(batch_indices)

        batch_indexMat_labels = indexMat_labels[batch_indices] # 使用下标查找，必须是ndarray类型类型
        indexMat_labels=np.delete(indexMat_labels,batch_indices,axis=0)
        height = indexMat_labels.shape[0]

        # 将数据分割成indexMat和labels
        batch_labels, batch_indexMat = create_word2vec.split_labels_indexMat(batch_indexMat_labels)
        # 是否进行独热编码
        if one_hot:
            batch_labels = batch_labels.reshape(len(batch_labels), 1)
            onehot_encoder = preprocessing.OneHotEncoder(sparse=False,categories=[range(labels_nums)])
            batch_labels = onehot_encoder.fit_transform(batch_labels)
        yield batch_indexMat,batch_labels


def get_next_batch(batch):
    return batch.__next__()

def get_file_list(file_dir,postfix):
    '''
    获得后缀名为postfix所有文件列表
    :param file_dir:
    :param postfix:
    :return:
    '''
    file_dir=os.path.join(file_dir,postfix)
    file_list=glob.glob(file_dir)
    file_list.sort()
    return file_list

def create_test_data(out_dir):
    '''
    产生测试数据
    :return:
    '''
    data1 = np.arange(0, 10)
    data1 = np.transpose(data1.reshape([2, 5]))
    label1 = np.arange(0, 5)
    label1 = label1.reshape([5, 1])

    path1 = os.path.join(out_dir,'data1.npy')
    indexMat1 = np.concatenate([label1, data1], axis=1)  # 矩阵拼接，第一列为labels
    np.save(path1, indexMat1)

    data2 = np.arange(15, 25)
    data2 = np.transpose(data2.reshape([2, 5]))
    label2 = np.arange(5, 10)
    label2 = label2.reshape([5, 1])

    path2 = os.path.join(out_dir,'data2.npy')
    indexMat2 = np.concatenate([label2, data2], axis=1)
    np.save(path2, indexMat2)

    data3 = np.arange(30, 40)
    data3 = np.transpose(data3.reshape([2, 5]))
    label3 = np.arange(10, 15)
    label3 = label3.reshape([5, 1])

    path3 = os.path.join(out_dir,'data3.npy')
    indexMat3 = np.concatenate([label3, data3], axis=1)
    np.save(path3, indexMat3)

    print('indexMat1:\n{}'.format(indexMat1))
    print('indexMat2:\n{}'.format(indexMat2))
    print('indexMat3:\n{}'.format(indexMat3))


if __name__ == '__main__':
    train_out_dir='./train_data'
    labels_file = 'test_label.txt'
    labels_set = files_processing.read_txt(labels_file)

    files_processing.info_labels_set(labels_set)

    # create_test_data(out_dir)
    file_list=get_file_list(file_dir=train_out_dir, postfix='*.npy')
    create_word2vec.info_npy(file_list)

    iter = 5  # 迭代3次，每次输出一个batch个
    labels_nums=len(labels_set)
    batch = get_data_batch(file_list, labels_nums=14,batch_size=6, shuffle=False,one_hot=False)
    for i in range(iter):
        print('**************************')
        batch_indexMat, batch_labels = get_next_batch(batch)

        # 解码：将索引矩阵解码为字词
        # word2vec_path = 'out/trained_word2vec.model'
        word2vec_path='out/THUCNews_word2vec300.model'
        w2vModel = create_word2vec.load_wordVectors(word2vec_path)
        # batch_indexMat = create_word2vec.indexMat2vector_lookup(w2vModel, batch_indexMat)

        # batch_indexMat = create_word2vec.indexMat2word(w2vModel,batch_indexMat)
        # batch_indexMat=np.asarray(batch_indexMat)

        # 解码：将int类型的labels解码为字符串类型的labels
        # batch_labels = batch_labels.flatten().tolist()
        # batch_labels=files_processing.labels_decoding(batch_labels,labels_set)

        # print('batch_images:shape:{}\n{}'.format(batch_indexMat.shape,batch_indexMat))
        print('batch_images:\n{}'.format(batch_indexMat))

        print('batch_labels:\n{}'.format(batch_labels))


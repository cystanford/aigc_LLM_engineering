# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : create_word2vec.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-08 17:37:21
"""
from gensim.models import Word2Vec
import random
import numpy as np
import os
import math
from utils import files_processing,segment


def info_npy(file_list):
    sizes=0
    for file in file_list:
        data = np.load(file)
        print("data.shape:{}".format(data.shape))
        size = data.shape[0]
        sizes+=size
    print("files nums:{}, data nums:{}".format(len(file_list), sizes))
    return sizes

def save_multi_file(files_list,labels_list,word2vec_path,out_dir,prefix,batchSize,max_sentence_length,labels_set=None,shuffle=False):
    '''
    将文件内容映射为索引矩阵，并且将数据保存为多个文件
    :param files_list:
    :param labels_list:
    :param word2vec_path: word2vec模型的位置
    :param out_dir: 文件保存的目录
    :param prefix:  保存文件的前缀名
    :param batchSize: 将多个文件内容保存为一个文件
    :param labels_set: labels集合
    :return:
    '''
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 把该目录下的所有文件都删除
    files_processing.delete_dir_file(out_dir)

    if shuffle:
        random.seed(100)
        random.shuffle(files_list)
        random.seed(100)
        random.shuffle(labels_list)

    sample_num = len(files_list)
    w2vModel=load_wordVectors(word2vec_path)
    if labels_set is None:
        labels_set= files_processing.get_labels_set(label_list)

    labels_list, labels_set = files_processing.labels_encoding(labels_list, labels_set)
    labels_list=labels_list.tolist()
    batchNum = int(math.ceil(1.0 * sample_num / batchSize))
    for i in range(batchNum):
        start = i * batchSize
        end = min((i + 1) * batchSize, sample_num)
        batch_files = files_list[start:end]
        batch_labels = labels_list[start:end]

        # 读取文件内容，字词分割
        batch_content = files_processing.read_files_list(batch_files, max_sentence_length,padding_token='<PAD>')
        # 将字词转为索引矩阵
        batch_indexMat = word2indexMat(w2vModel, batch_content, max_sentence_length)
        batch_labels=np.asarray(batch_labels)
        batch_labels = batch_labels.reshape([len(batch_labels), 1])

        # 保存*.npy文件
        filename = os.path.join(out_dir,prefix + '{0}.npy'.format(i))
        labels_indexMat = cat_labels_indexMat(batch_labels, batch_indexMat)
        np.save(filename, labels_indexMat)
        print('step:{}/{}, save:{}, data.shape{}'.format(i,batchNum,filename,labels_indexMat.shape))


def cat_labels_indexMat(labels,indexMat):
    indexMat_labels = np.concatenate([labels,indexMat], axis=1)
    return indexMat_labels

def split_labels_indexMat(indexMat_labels,label_index=0):
    labels = indexMat_labels[:, 0:label_index+1]     # 第一列是labels
    indexMat = indexMat_labels[:, label_index+1:]  # 其余是indexMat
    return labels, indexMat

def load_wordVectors(word2vec_path):
    w2vModel = Word2Vec.load(word2vec_path)
    return w2vModel

def word2vector_lookup(w2vModel, sentences):
    '''
    将字词转换为词向量
    :param w2vModel: word2vector模型
    :param sentences: type->list[list[str]]
    :return: sentences对应的词向量,type->list[list[ndarray[list]]
    '''
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                v=w2vModel[word]
                this_vector.append(v)
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    all_vectors=np.array(all_vectors)
    return all_vectors

def word2indexMat(w2vModel, sentences, max_sentence_length):
    '''
    将字词word转为索引矩阵
    :param w2vModel:
    :param sentences:
    :param max_sentence_length:
    :return:
    '''
    nums_sample=len(sentences)
    indexMat = np.zeros((nums_sample, max_sentence_length), dtype='int32')
    rows = 0
    for sentence in sentences:
        indexCounter = 0
        for word in sentence:
            try:
                index = w2vModel.wv.vocab[word].index  # 获得单词word的下标
                indexMat[rows][indexCounter] = index
            except :
                indexMat[rows][indexCounter] = 0  # Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= max_sentence_length:
                break
        rows+=1
    return indexMat

def indexMat2word(w2vModel, indexMat, max_sentence_length=None):
    '''
    将索引矩阵转为字词word
    :param w2vModel:
    :param indexMat:
    :param max_sentence_length:
    :return:
    '''
    if max_sentence_length is None:
        row,col =indexMat.shape
        max_sentence_length=col
    sentences=[]
    for Mat in indexMat:
        indexCounter = 0
        sentence=[]
        for index in Mat:
            try:
                word = w2vModel.wv.index2word[index] # 获得单词word的下标
                sentence+=[word]
            except :
                sentence+=['<PAD>']
            indexCounter = indexCounter + 1
            if indexCounter >= max_sentence_length:
                break
        sentences.append(sentence)
    return sentences

def save_indexMat(indexMat,path):
    np.save(path, indexMat)

def load_indexMat(path):
    indexMat = np.load(path)
    return indexMat

def indexMat2vector_lookup(w2vModel,indexMat):
    '''
    将索引矩阵转为词向量
    :param w2vModel:
    :param indexMat:
    :return: 词向量
    '''
    all_vectors = w2vModel.wv.vectors[indexMat]
    return all_vectors

def pos_neg_test():
    positive_data_file = "./data/ham_5000.utf8"
    negative_data_file = './data/spam_5000.utf8'

    word2vec_path = 'out/trained_word2vec.model'
    sentences, labels = files_processing.load_pos_neg_files(positive_data_file, negative_data_file)
    # embedding_test(positive_data_file,negative_data_file)
    sentences, max_document_length = segment.padding_sentences(sentences, '<PADDING>', padding_sentence_length=190)
    # train_wordVectors(sentences,embedding_size=128,word2vec_path=word2vec_path) # 训练word2vec，并保存word2vec_path
    w2vModel=load_wordVectors(word2vec_path) #加载训练好的word2vec模型

    '''
    转换词向量提供有两种方法：
    [1]直接转换：根据字词直接映射到词向量：word2vector_lookup
    [2]间接转换：先将字词转为索引矩阵，再由索引矩阵映射到词向量：word2indexMat->indexMat2vector_lookup
    '''
    # [1]根据字词直接映射到词向量
    x1=word2vector_lookup(w2vModel, sentences)

    # [2]先将字词转为索引矩阵，再由索引矩阵映射到词向量
    indexMat_path = 'out/indexMat.npy'
    indexMat=word2indexMat(w2vModel, sentences, max_sentence_length=190) # 将字词转为索引矩阵
    save_indexMat(indexMat, indexMat_path)
    x2=indexMat2vector_lookup(w2vModel, indexMat) # 索引矩阵映射到词向量
    print("x.shape = {}".format(x2.shape))# shape=(10000, 190, 128)->(样本个数10000,每个样本的字词个数190，每个字词的向量长度128)

if __name__=='__main__':
    THUCNews_path='/home/ubuntu/project/tfTest/THUCNews/test'
    # THUCNews_path='/home/ubuntu/project/tfTest/THUCNews/spam'
    # THUCNews_path='/home/ubuntu/project/tfTest/THUCNews/THUCNews'
    # 读取所有文件列表
    files_list, label_list = files_processing.gen_files_labels(THUCNews_path)

    max_sentence_length=300
    word2vec_path="../data/THUCNews_word2vec300.model"

    # 获得标签集合，并保存在本地
    # labels_set=['星座','财经','教育']
    # labels_set = files_processing.get_labels_set(label_list)
    labels_file='../data/THUCNews_labels.txt'
    # files_processing.write_txt(labels_file, labels_set)

    # 将数据划分为train val数据集
    train_files, train_label, val_files, val_label= files_processing.split_train_val_list(files_list, label_list, facror=0.9, shuffle=True)

    # contents, labels=files_processing.read_files_labels(files_list,label_list)
    # word2vec_path = 'out/trained_word2vec.model'
    train_out_dir='../data/train_data'
    prefix='train_data'
    batchSize=20000
    labels_set=files_processing.read_txt(labels_file)
    # labels_set2 = files_processing.read_txt(labels_file)
    save_multi_file(files_list=train_files,
                    labels_list=train_label,
                    word2vec_path=word2vec_path,
                    out_dir=train_out_dir,
                    prefix=prefix,
                    batchSize=batchSize,
                    max_sentence_length=max_sentence_length,
                    labels_set=labels_set,
                    shuffle=True)
    print("*******************************************************")
    val_out_dir='../data/val_data'
    prefix='val_data'
    save_multi_file(files_list=val_files,
                    labels_list=val_label,
                    word2vec_path=word2vec_path,
                    out_dir=val_out_dir,
                    prefix=prefix,
                    batchSize=batchSize,
                    max_sentence_length=max_sentence_length,
                    labels_set=labels_set,
                    shuffle=True)
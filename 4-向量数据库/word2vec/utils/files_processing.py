# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : files_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-08 19:57:42
"""
import random
import numpy as np
import pandas as pd
import os
import io
import re
import os
import math
from sklearn import preprocessing
from utils import segment
import pickle
import glob

def split_train_val_array(data,labels,facror=0.6,shuffle=True):
    '''

    :param data:
    :param labels:
    :param facror:
    :param shuffle:
    :return:
    '''
    indices = list(range(len(labels)))
    if shuffle:
        random.shuffle(indices)
    split=int(len(labels)*facror)
    train_data_index=indices[:split]
    val_data_index=indices[split:]

    # 训练数据
    train_data=data[train_data_index]
    train_label=labels[train_data_index]

    # 测数数据
    val_data=data[val_data_index]
    val_label=labels[val_data_index]
    return train_data,train_label,val_data,val_label

def split_train_val_list(data_list,labels_list,facror=0.6,shuffle=True):
    '''

    :param data:
    :param labels:
    :param facror:
    :param shuffle:
    :return:
    '''
    if shuffle:
        random.seed(100)
        random.shuffle(data_list)
        random.seed(100)
        random.shuffle(labels_list)
    split=int(len(labels_list)*facror)
    # 训练数据
    train_data=data_list[:split]
    train_label=labels_list[:split]
    # 测数数据
    val_data=data_list[split:]
    val_label=labels_list[split:]

    print("train_data:{},train_label:{}".format(len(train_data),len(train_label)))
    print("val_data  :{},val_label  :{}".format(len(val_data),len(val_label)))

    return train_data,train_label,val_data,val_label


def load_pos_neg_files(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = read_and_clean_zh_file(positive_data_file)
    negative_examples = read_and_clean_zh_file(negative_data_file)
    # Combine data
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    x_text = [sentence.split(' ') for sentence in x_text]
    return [x_text, y]

def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param file_dir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def get_files_list(file_dir,postfix='ALL'):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    '''
    postfix=postfix.split('.')[-1]
    file_list=[]
    filePath_list = getFilePathList(file_dir)
    if postfix=='ALL':
        file_list=filePath_list
    else:
        for file in filePath_list:
            basename=os.path.basename(file)  # 获得路径下的文件名
            postfix_name=basename.split('.')[-1]
            if postfix_name==postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

def gen_files_labels(files_dir):
    '''
    获取files_dir路径下所有文件路径，以及labels,其中labels用子级文件名表示
    files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
    :param files_dir:
    :return:filePath_list所有文件的路径,label_list对应的labels
    '''
    filePath_list = getFilePathList(files_dir)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set=list(set(label_list))
    print("labels:{}".format(labels_set))

    # 标签统计计数
    print(pd.value_counts(label_list))
    return filePath_list,label_list

def read_files_list(files_list,max_sentence_length,padding_token='<PAN>'):
    content_list = segment.segment_files_list(files_list, stopwords=[], segment_type='word')
    content_list = segment.padding_sentences(content_list,
                                             padding_token=padding_token,
                                             padding_sentence_length=max_sentence_length)
    return content_list


def get_labels_set(label_list):
    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))
    return labels_set

def labels_encoding(label_list,labels_set=None):
    '''
    将字符串类型的label编码成int,-1表示未知的labels
    :param label_list:
    :return:
    '''
    # 将labels转为整数编码
    if labels_set is None:
        labels_set=list(set(label_list))

    labels=[]
    for label in  label_list:
        if label in labels_set:
            k=labels_set.index(label)
            labels+=[k]
        else:
            print("warning unknow label")
            labels+=[-1] # -1表示未知的labels,unknow

    labels = np.asarray(labels)
    # 也可以用下面的方法：将labels转为整数编码
    # labelEncoder = preprocessing.LabelEncoder()
    # labels = labelEncoder.fit_transform(label_list)
    # labels_set = labelEncoder.classes_
    for i in range(len(labels_set)):
        print("labels:{}->{}".format(labels_set[i],i))

    return labels,labels_set

def labels_decoding(labels,labels_set):
    '''
    将int类型的label解码成字符串类型的label
    :param label_list:
    :return:
    '''
    for i in range(len(labels_set)):
        print("labels:{}->{}".format(labels_set[i],i))
    labels_list=[]
    for i in labels:
        if i ==-1:
            print("warning unknow label")
            labels_list.append('unknow')
            continue
        labels_list.append(labels_set[i])
    return labels_list

def read_and_clean_zh_file(input_file, output_cleaned_file = None):
    lines = list(open(input_file, "rb").readlines())
    lines = [segment.clean_str(segment.seperate_line(line.decode('utf-8'))) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines

def delete_dir_file(dir_path):
    ls = os.listdir(dir_path)
    for i in ls:
        c_path = os.path.join(dir_path, i)
        if os.path.isdir(c_path):
            delete_dir_file(c_path)
        else:
            os.remove(c_path)

def write_txt(file_name,content_list,mode="w"):
    with open(file_name,mode) as f:
        for line in content_list:
            f.write(line+"\n")

def read_txt(file_name):
    content_list=[]
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            content_list.append(line)
    return content_list

def save_data(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def info_labels_set(labels_set):
    for i in range(len(labels_set)):
        print("labels:{}->{}".format(labels_set[i],i))

if __name__=="__main__":
    # THUCNews_path = "D:/tensorflow/nlp-learning-tutorials/THUCNews/data"
    THUCNews_path='/home/ubuntu/project/tfTest/THUCNews/test'
    files_list, label_list = gen_files_labels(THUCNews_path)
    print('sample_size:{}'.format(len(files_list)))
    labels_set=['星座','财经','教育']
    label_list=['星座','AA','财经','教育']
    # labels_set=get_labels_set(label_list)
    labels_list, labels_set=labels_encoding(label_list,labels_set)
    labels_list=labels_decoding(labels_list, labels_set)

    contents_list=read_files_list(files_list, max_sentence_length=190,padding_token='<PAD>')

# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : segment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:51:53
"""

##
import jieba
import os
import io
import math
import re
from utils import files_processing

'''
read() 每次读取整个文件，它通常将读取到底文件内容放到一个字符串变量中，也就是说 .read() 生成文件内容是一个字符串类型。
readline()每只读取文件的一行，通常也是读取到的一行内容放到一个字符串变量中，返回str类型。
readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。
'''
def load_stopWords(path):
    '''
    加载停用词
    :param path:
    :return:
    '''
    stopwords = []
    with open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

def common_stopwords():
    '''
    常用的停用词
    :return:
    '''
    Stopwords=[]
    # Stopwords=['\n','',' ','  ','\n\n']
    Stopwords=['\n','',' ','\n\n']
    return Stopwords

def padding_sentence(sentence, padding_token, padding_sentence_length):
    '''
    padding句子长度
    :param sentence: type->list[str]
    :param padding_token:
    :param padding_sentence_length:
    :return:
    '''
    if len(sentence) > padding_sentence_length:
        sentence = sentence[:padding_sentence_length]
    else:
        sentence.extend([padding_token] * (padding_sentence_length - len(sentence)))
    return sentence

def padding_sentences(sentences_list, padding_token, padding_sentence_length):
    '''
    padding句子长度
    :param sentences_list: type->list[list[str]]
    :param padding_token:  设置padding的内容
    :param padding_sentence_length: padding的长度
    :return:
    '''
    for i, sentence in enumerate(sentences_list):
        sentence=padding_sentence(sentence, padding_token, padding_sentence_length)
        sentences_list[i]=sentence
    return sentences_list

def read_file_content(file,mode='r'):
    '''
    读取文件内容，并去除去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    :param file:
    :param mode:
    :return: str
    '''
    with open(file, mode=mode) as f:
        lines = f.readlines()
        contents=[]
        for line in lines:
            line=line.strip()
            if line.rstrip()!='':
                contents.append(line)
        contents='\n'.join(contents)
    return contents

def read_files_list_content(files_list,mode='r'):
    '''
    读取文件列表内容，并去除去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    :param files_list: 文件列表
    :param mode:
    :return: list[str]
    '''
    content_list=[]
    for i , file in enumerate(files_list):
        content=read_file_content(file,mode=mode)
        content_list.append(content)
    return content_list


def save_content(file,content,mode='wb'):
    with open(file, mode=mode) as f:
        content = content.encode('utf-8')
        f.write(content)

def save_content_list(file,content_list,mode='wb'):
    for i , con_list in enumerate(content_list):
        content = ' '.join(con_list)
        content+='\n'
        save_content(file, content, mode=mode)

def cut_content_jieba(content):
    '''
    按字词word进行分割
    :param content: str
    :return:
    '''
    lines_cut = jieba.cut(content)
    return lines_cut

def cut_content_char(content):
    '''
    按字符char进行分割
    :param content: str
    :return:
    '''
    lines_cut = clean_str(seperate_line(content))
    return lines_cut


def delete_stopwords(lines_list, stopwords= []):
    sentence_segment=[]
    for word in lines_list:
        if word not in stopwords:
            sentence_segment.append(word)
    return sentence_segment

def segment_content_word(content,stopwords=[]):
    lines_cut_list=cut_content_jieba(content)
    segment_list=delete_stopwords(lines_cut_list,stopwords)
    return segment_list

def segment_content_char(content,stopwords=[]):
    lines_cut_str=cut_content_char(content)
    lines_cut_list = lines_cut_str.split(' ')
    segment_list=delete_stopwords(lines_cut_list,stopwords)
    return segment_list

def segment_file(file, stopwords=[], segment_type='word'):
    '''
    字词分割
    :param file:
    :param stopwords:
    :param segment_type: word or char，选择分割类型，按照字符char，还是字词word分割
    :return:
    '''
    content = read_file_content(file, mode='r')
    if segment_type=='word' or segment_type is None:
        segment_content = segment_content_word(content, stopwords)
    elif segment_type=='char':
        segment_content = segment_content_char(content, stopwords)
    return segment_content

def segment_files_list(files_list,stopwords=[],segment_type='word'):
    '''
    字词分割
    :param files_list:
    :param stopwords:
    :param segment_type: word or char，选择分割类型，按照字符char，还是字词word分割
    :return:
    '''
    content_list=[]
    for i, file in enumerate(files_list):
        segment_content=segment_file(file,stopwords,segment_type)
        content_list.append(segment_content)
    return content_list

def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()

def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def combine_files_content(file_list,out_file):
    '''
    合并输出一个文件
    :param file_list:
    :param out_file:
    :return:
    '''
    f2 = open(out_file, 'wb')
    for i, file in enumerate(file_list):
        with io.open(file, encoding='utf8') as file:
            lines = file.readlines()
            lines = ''.join(lines)
        result = ' '.join(lines)
        result+='\n'
        result = result.encode('utf-8')
        f2.write(result)
    f2.close()


def batch_processing_files(files_list,segment_out_dir,batchSize,stopwords=[]):
    '''
    批量分割文件字词，并将batchSize的文件合并一个文件
    :param files_list: 文件列表
    :param segment_out_dir: 字符分割文件输出的目录
    :param batchSize:
    :param stopwords: 停用词
    :return:
    '''
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)
    files_processing.delete_dir_file(segment_out_dir)

    sample_num=len(files_list)
    batchNum = int(math.ceil(1.0 * sample_num / batchSize))
    for i in range(batchNum):
        segment_out_name = os.path.join(segment_out_dir, 'segment_{}.txt'.format(i))
        start = i * batchSize
        end = min((i + 1) * batchSize, sample_num)
        batch_files = files_list[start:end]
        content_list=segment_files_list(batch_files, stopwords,segment_type='word')
        # content_list=padding_sentences(content_list, padding_token='<PAD>', padding_sentence_length=15)
        save_content_list(segment_out_name,content_list,mode='ab')
        print("segment files:{}".format(segment_out_name))


if __name__=='__main__':
    # 多线程分词
    # jieba.enable_parallel()
    # 加载自定义词典
    # user_path = '../data/user_dict.txt'
    # jieba.load_userdict(user_path)

    # stopwords_path='data/stop_words.txt'
    # stopwords=load_stopwords(stopwords_path)
    stopwords=common_stopwords()

    # file_dir='../data/source2'
    file_dir='/home/ubuntu/project/tfTest/THUCNews/THUCNews'

    segment_out_dir='../data/segment'
    files_list=files_processing.get_files_list(file_dir,postfix='*.txt')

    # segment_out_dir='data/segment_conbine.txt'
    # combine_files_content(files_list, segment_out_dir,stopwords)
    batch_processing_files(files_list, segment_out_dir, batchSize=1000, stopwords=[])


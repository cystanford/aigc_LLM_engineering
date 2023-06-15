import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
# 512维，data包含2000个向量，每个向量符合正态分布
d = 512          
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')
# print(data[0])
print(data.shape)
# 查看第6个向量是不是符合正态分布
import matplotlib.pyplot as plt 
plt.hist(data[5])
plt.show()

# 精确索引
query = []
n_query = 10
mu = 3
sigma = 0.1
np.random.seed(12) 
query = []
for i in range(n_query):
    query.append(np.random.normal(mu, sigma, d))
query = np.array(query).astype('float32')

import faiss
index = faiss.IndexFlatL2(d)  # 构建 IndexFlatL2
print(index.is_trained)  # False时需要train
index.add(data)  #添加数据
print(index.ntotal)  #index中向量的个数

#精确索引无需训练便可直接查询
k = 10  # 返回结果个数
query_self = data[:5]  # 查询本身
dis, ind = index.search(query_self, k)
print(dis.shape) # 打印张量 (5, 10)
print(ind.shape) # 打印张量 (5, 10)
print(dis)  # 升序返回每个查询向量的距离
print(ind)  # 升序返回每个查询向量


# 倒排表快速索引
nlist = 50  # 将数据库向量分割为多少了维诺空间
k = 10
quantizer = faiss.IndexFlatL2(d)  # 量化器
# METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
print(index.is_trained)
#倒排表索引类型需要训练, 训练数据集应该与数据库数据集同分布
index.train(data)
print(index.is_trained)

index.add(data)
index.nprobe = 50  # 选择n个维诺空间进行索引,
#dis, ind = index.search(query, k)
dis, ind = index.search(query_self, k)
print(dis)
print(ind)


# 乘积量化索引
nlist = 50
m = 8  # 列方向划分个数，必须能被d整除
k = 10
quantizer = faiss.IndexFlatL2(d)  
# 8 表示每个子向量被编码为 8 bits
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) 
index.train(data)
index.add(data)
index.nprobe = 50
dis, ind = index.search(query_self, k)  # 查询自身
print(dis)
print(ind)
"""
dis, ind = index.search(query, k)  # 真实查询
print(dis)
print(ind)
"""
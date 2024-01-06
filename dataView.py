import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

DirPath = r"./Citeseer/citeseer"
Cites = 'citeseer.cites'
Content = 'citeseer.content'

# DirPath = r"D:\MyFile\DataSet\GraphNN\cora"
# Cites = 'cora.cites'
# Content = 'cora.content'

if __name__ == "__main__":
    raw_data = pd.read_csv(os.path.join(DirPath, Content), sep='\t', header=None, dtype={0: str})
    print(raw_data.head())

    # 节点数量
    num_nodes = raw_data.shape[0]
    print("node nums：", num_nodes)

    print("content shape: ", raw_data.shape)

    raw_data_cites = pd.read_csv(os.path.join(DirPath, Cites), sep='\t', header=None)
    print("cites shape: ", raw_data_cites.shape)
    # print(raw_data_cites)

    features = raw_data.iloc[:, 1:-1]
    print("feature shape: ", features.shape)

    # one-hot encoding
    # labels = pd.get_dummies(raw_data[raw_data.shape[1] - 1])
    # print("labels shape: ", labels.shape)

    # Convert labels to integers
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(raw_data[raw_data.shape[1] - 1])
    print("labels shape: ", labels.shape)

    # 将节点重新编号为
    index_id = list(raw_data.index)
    paper_id = list(raw_data[0])
    paper_id = [str(i) for i in paper_id]
    # 构建映射字典
    print(index_id)
    Map = dict(zip(paper_id, index_id))
    # print(Map)

    # 根据节点个数定义矩阵维度
    matrix = np.zeros((num_nodes, num_nodes))

    # 根据边构建矩阵
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        if str(i) in Map.keys() and str(j) in Map.keys():
            x = Map[str(i)]
            y = Map[str(j)]
            matrix[x][y] = matrix[y][x] = 1  # 无向图：有引用关系的样本点之间取1

    # 查看邻接矩阵的元素
    print("matrix shape: ", matrix.shape)

    features = np.array(features)
    labels = np.array(labels)
    adj = np.array(matrix)

    print(features, labels, adj)
    pass

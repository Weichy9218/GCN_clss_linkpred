import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import networkx as nx
import numpy as np


class CoraDataset(Dataset):
    def __init__(self, Edge_list_path, Node_features_path):
        self.features, self.labels, self.adjacency_matrix = self.load_data(Edge_list_path, Node_features_path)
        self.num_nodes = len(self.labels)

    @staticmethod
    def load_data(Edge_list_path, Node_features_path):
        # Load Cora dataset using NetworkX as an example
        Gnx = nx.read_edgelist(Edge_list_path, nodetype=int)
        # print(Gnx)

        # Load node features and labels
        feature_names = ["w_{}".format(ii) for ii in range(1433)]
        column_names = feature_names + ["subject"]
        
        node_data = pd.read_csv(Node_features_path, sep='\t', header=None, names=column_names)
        node_features = node_data.to_dict("index")

        nx.set_node_attributes(Gnx, node_features)

        # 提取节点特征
        Features = np.array([Gnx.nodes[i]["w_{}".format(ii)] for i in Gnx.nodes for ii in range(1433)])
        Features = Features.reshape(len(Gnx.nodes), -1)

        # label转换成0123456 index
        labels = np.array([Gnx.nodes[i]["subject"] for i in Gnx.nodes])
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)

        # 从图中提取邻接矩阵 sparse SciPy matrix
        adjacency_matrix = nx.adjacency_matrix(Gnx)
        
        return Features, labels, adjacency_matrix

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        Features = torch.FloatTensor(self.features[idx])
        Label = torch.LongTensor([self.labels[idx]])
        # print(Features, Label)
        # Extract a row from the sparse matrix and convert it to dense
        adjacency_row = self.adjacency_matrix[:, [idx]].toarray().flatten()
        Adjacency = torch.FloatTensor(adjacency_row)

        return Features, Label, Adjacency


if __name__ == "__main__":
    # Example usage
    DirPath = r"./data/cora"
    edge_list_path = os.path.join(DirPath, 'cora.cites')
    node_features_path = os.path.join(DirPath, 'cora.content')

    cora_dataset = CoraDataset(edge_list_path, node_features_path)

    features, label, adjacency = cora_dataset[0]
    print("Sample Features:", features)
    print("Sample Label:", label)
    print("Sample Adjacency Matrix:", adjacency)
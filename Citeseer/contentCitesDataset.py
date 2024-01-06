import os
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np


class ContentCitesDataset(Dataset):
    def __init__(self, DirPath=r"./Citeseer/citeseer",
                 Cites='citeseer.cites', Content='citeseer.content'):
        self.DirPath = DirPath
        self.Content = Content
        self.Cites = Cites
        self.features, self.labels, self.dense_matrix = self.load_data()

    def load_data(self):
        raw_data = pd.read_csv(os.path.join(self.DirPath, self.Content), sep='\t', header=None, dtype={0: str})
        raw_data_cites = pd.read_csv(os.path.join(self.DirPath, self.Cites), sep='\t', header=None)
        num_nodes = raw_data.shape[0]
        w_features = raw_data.iloc[:, 1:-1]
        le = preprocessing.LabelEncoder()
        subject_labels = le.fit_transform(raw_data[raw_data.shape[1] - 1])

        index_id = list(raw_data.index)
        paper_id = list(raw_data[0])
        paper_id = [str(i) for i in paper_id]

        idMap = dict(zip(paper_id, index_id))

        Dense_matrix = np.zeros((num_nodes, num_nodes))

        for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
            if str(i) in idMap.keys() and str(j) in idMap.keys():
                x = idMap[str(i)]
                y = idMap[str(j)]
                Dense_matrix[x][y] = Dense_matrix[y][x] = 1  # 无向图：有引用关系的样本点之间取1

        w_features = np.array(w_features)
        subject_labels = np.array(subject_labels)
        Dense_matrix = np.array(Dense_matrix)

        return w_features, subject_labels, Dense_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        Features = torch.FloatTensor(self.features[idx])
        Label = torch.LongTensor([self.labels[idx]])
        # print(Features, Label)

        adjacency_row = self.dense_matrix[idx, :].flatten()
        Adjacency = torch.FloatTensor(adjacency_row)

        return Features, Label, Adjacency


if __name__ == "__main__":
    # Example usage
    # DirPath = r"D:\MyFile\DataSet\GraphNN\citeseer",
    # Cites = 'citeseer.cites'
    # Content = 'citeseer.content'
    cora_dataset = ContentCitesDataset()

    features, labels, dense_matrix = cora_dataset.load_data()
    print(features, labels, dense_matrix)
    # 使用 Counter 统计元素个数
    label_counts = Counter(labels)

    # 打印结果
    for label, count in label_counts.items():
        print(f"Label {label}: {count} occurrences")

    sample_idx = 0
    features, label, adjacency = cora_dataset[sample_idx]
    print("Sample Features:", features)
    print("Sample Label:", label)
    print("Sample Adjacency Matrix:", adjacency)

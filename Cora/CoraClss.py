import os

from torch import optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from Cora.CoraDataset import CoraDataset


class GCNLayer(nn.Module):
    def __init__(self, Input_dim, Output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(Input_dim, Output_dim)

    def forward(self, Adjacency, Features):
        # GCN layer operation with symmetric normalization
        normalized_adjacency = self.normalize_adjacency(Adjacency)

        # GCN layer operation
        Output = torch.spmm(normalized_adjacency, Features)  # 仅仅考虑相邻点
        Output = self.linear(Output)
        return Output

    @staticmethod
    def normalize_adjacency(Adjacency):
        normalization = torch.pow(Adjacency.sum(dim=1), -0.5)
        normalization[torch.isinf(normalization)] = 0.0  # Handle division by zero
        normalized_adjacency = Adjacency * normalization.unsqueeze(1) * normalization.unsqueeze(0)
        return normalized_adjacency


class GCNModel(nn.Module):
    def __init__(self, Input_dim, Hidden_dim, Output_dim):
        super(GCNModel, self).__init__()
        self.layer1 = GCNLayer(Input_dim, Hidden_dim)
        self.layer2 = GCNLayer(Hidden_dim, Hidden_dim)
        self.layer3 = GCNLayer(Hidden_dim, Output_dim)
        self.dropout = torch.nn.Dropout(p=0.01)

    def forward(self, Adjacency, Features):
        # Forward pass through GCN layers
        hidden = self.layer1(Adjacency, Features)
        hidden = self.PairNorm(hidden)  # pairNorm
        hidden = F.relu(hidden)
        # hidden = F.sigmoid(hidden)
        hidden = self.dropout(hidden)
        Output = self.layer3(Adjacency, hidden)
        return Output
    

    
    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature


def init_seeds(seed=1234):
    # 随机seed,保证实验可重复性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

if __name__ == "__main__":

    init_seeds(seed=1234)  # 921
    # 使用你的数据集维度初始化模型
    num_epochs = 200
    input_dim = 1433  # 输入特征的维度，例如1433
    hidden_dim = 256  # 隐藏层维度，可以根据需要调整
    output_dim = 7  # 输出维度，例如类别数量
    selfloop = True  # 自环
    # num_layers = 3   # 层数 至少为2层
    # dropout = 0.1
    learning_rate = 1e-3

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNModel(Input_dim=input_dim, Hidden_dim=hidden_dim, Output_dim=output_dim) #  num_layers=num_layers, dropout=dropout
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 加载数据集
    DirPath = r"./Cora/cora"
    edge_list_path = os.path.join(DirPath, 'cora.cites')
    node_features_path = os.path.join(DirPath, 'cora.content')

    cora_dataset = CoraDataset(edge_list_path, node_features_path)

    # 划分数据集
    total_size = len(cora_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 设置训练、验证和测试的数量
    idx_train = range(train_size)
    idx_val = range(train_size, train_size + val_size)
    idx_test = range(total_size - test_size, total_size)

    Batch_Size = len(cora_dataset)
    dataloader = DataLoader(cora_dataset, batch_size=Batch_Size, shuffle=False)

    # 训练模型
    best_val_loss = float('inf')
    best_model_state_dict = None

    # 初始化记录训练损失和验证准确率的列表
    train_losses = []
    val_accuracies = []

    for features, labels, adjacency in dataloader:
        model.train()

        if selfloop:
            num_nodes = adjacency.size(0)
            identity_matrix = torch.eye(num_nodes)
            adjacency = adjacency + identity_matrix

        features, labels, adjacency = features.to(device), labels.to(device), adjacency.to(device)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(adjacency, features)
            train_loss = criterion(output[idx_train], labels.squeeze()[idx_train])
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
           
            # 验证模型
            model.eval()
            with torch.no_grad():
                output = model(adjacency, features)
                val_loss = criterion(output[idx_val], labels.squeeze()[idx_val])

            _, predicted_val = torch.max(output, 1)
            # 计算并记录验证准确率
            correct = (predicted_val[idx_val] == labels.squeeze()[idx_val]).sum().item()
            val_accuracy = correct / len(idx_val)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

            # 保存最佳模型参数
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()


        # 选择最佳模型进行测试
        model.load_state_dict(best_model_state_dict)
        model.eval()

        # 在测试集上进行评估
        with torch.no_grad():
            output = model(adjacency, features)
            test_loss = criterion(output[idx_test], labels.squeeze()[idx_test]).item()
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == labels.squeeze()).sum().item()
   
        print(f"Test Loss: {test_loss}, Accuracy: {accuracy / len(labels)}")
    
    # 绘制训练损失和验证准确率图
    fig, ax1 = plt.subplots(figsize=(10, 8))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.plot(train_losses, label='Train Loss', color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # 实例化一个第二个y轴
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy', color=color)
    ax2.tick_params(axis='y')

    # fig.tight_layout()  # 调整布局
    plt.title("Training Loss and Validation Accuracy")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig('./fig/cora_class_pairnorm.png')  # 保存图像
    plt.show()
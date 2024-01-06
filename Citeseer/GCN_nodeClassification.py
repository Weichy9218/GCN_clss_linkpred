import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import sys
sys.path.append("..")
sys.path.append(".")
from Citeseer.contentCitesDataset import ContentCitesDataset


class GCNLayer(nn.Module):
    def __init__(self, Input_dim, Output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(Input_dim, Output_dim)

    def forward(self, Adjacency, Features):
        # GCN layer operation with symmetric normalization
        normalized_adjacency = self.symmetric_normalization(Adjacency)

        # GCN layer operation
        Output = torch.matmul(normalized_adjacency, Features)  # 仅仅考虑相邻点
        Output = self.linear(Output)
        return Output

    @staticmethod
    def symmetric_normalization(adjacency):
        """
        对称归一化：D^(-1/2) * A * D^(-1/2)，其中 D 是度矩阵，A 是邻接矩阵
        :param adjacency: 输入的邻接矩阵
        :return: normalized_adjacency: 归一化后的邻接矩阵
        """
        # 计算度矩阵 D 中每个节点的度
        degree = torch.sum(adjacency, dim=1)
        # 计算度矩阵的逆平方根，处理除以零的情况
        degree_inv_sqrt = 1.0 / torch.sqrt(degree + 1e-8)
        # 构建对称归一化的邻接矩阵
        normalized_adjacency = adjacency * degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0)
        return normalized_adjacency

    @staticmethod
    def random_walk_normalization(adjacency):
        """
        随机游走归一化：D^(-1) * A，其中 D 是度矩阵，A 是邻接矩阵
        :param adjacency: 输入的邻接矩阵
        :return: normalized_adjacency: 归一化后的邻接矩阵
        """
        # 计算度矩阵 D 中每个节点的度
        out_degree = torch.sum(adjacency, dim=1)
        # 计算度矩阵的逆，处理除以零的情况
        out_degree_inv = 1.0 / (out_degree + 1e-8)
        # 构建随机游走归一化的邻接矩阵
        normalized_adjacency = adjacency * out_degree_inv.unsqueeze(1)
        return normalized_adjacency



class GCNModel(nn.Module):
    def __init__(self, Input_dim, Hidden_dim, Output_dim):
        super(GCNModel, self).__init__()
        self.layer1 = GCNLayer(Input_dim, Hidden_dim)
        self.layer2 = GCNLayer(Hidden_dim, Hidden_dim)
        self.layer3 = GCNLayer(Hidden_dim, Output_dim)
        self.dropout = torch.nn.Dropout(p=0.05)

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


def init_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

def drop_edge(adjacency, dropEdge_rate):
    """
    DropEdge操作：以概率 dropEdge_rate 随机删除边
    :param adjacency: 输入的邻接矩阵
    :param dropEdge_rate: 删除边的概率
    :return: modified_adjacency: 经过DropEdge操作后的邻接矩阵
    """
    # 生成DropEdge的二值掩码
    mask = (torch.rand(adjacency.size(), device=adjacency.device) >= dropEdge_rate).float()
    # 对邻接矩阵应用DropEdge
    modified_adjacency = adjacency * mask

    return modified_adjacency


if __name__ == '__main__':
    
    citeseer_dataset = ContentCitesDataset(DirPath = r"./Citeseer/citeseer",
                                           Cites='citeseer.cites',
                                           Content='citeseer.content')

    features, labels, dense_matrix = citeseer_dataset.load_data()

    init_seed(222)

    index = torch.randperm(len(labels))

    # 定义划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # 计算划分点
    num_samples = len(labels)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)

    # 划分索引
    idx_train = index[:train_size]
    idx_val = index[train_size:(train_size + val_size)]
    idx_test = index[(train_size + val_size):]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    input_dim = features.shape[1]  # 输入特征的维度，例如1433
    hidden_dim = 256  # 隐藏层维度，可以根据需要调整
    output_dim = len(np.unique(labels))  # 输出维度，例如类别数量
    
    num_epochs = 100
    dropEdge_rate = 0
    selfloop = True

    model = GCNModel(input_dim, hidden_dim, output_dim)
    # Move model to GPU
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    features = torch.tensor(features, dtype=torch.float32).to(device)
    # Normalize features along the last dimension
    features = F.normalize(features, p=2, dim=-1)
    labels = torch.tensor(labels, dtype=torch.long).to(device).unsqueeze(1)
    dense_matrix = torch.tensor(dense_matrix, dtype=torch.float32).to(device)
    # Apply DropEdge
    if dropEdge_rate - 1e-6 > 0:
        dense_matrix = drop_edge(dense_matrix, dropEdge_rate)

    # 增加自环
    if selfloop:
        identity_matrix = torch.eye(dense_matrix.size(0), device=dense_matrix.device)
        dense_matrix = dense_matrix + identity_matrix

    best_val_loss = float('inf')
    best_model_state_dict = None

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):

        model.train()

        optimizer.zero_grad()
        output = model(dense_matrix, features)

        train_loss = criterion(output[idx_train], labels.squeeze()[idx_train])
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(dense_matrix, features)
            val_loss = criterion(output[idx_val], labels.squeeze()[idx_val])

            _, predicted_val = torch.max(output, 1)
            # 计算并记录验证准确率
            correct = (predicted_val[idx_val] == labels.squeeze()[idx_val]).sum().item()
            val_accuracy = correct / len(idx_val)
            val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        # 保存最佳模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()

    # 选择最佳模型进行测试
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # 在测试集上进行评估
    correct_predictions = 0
    with torch.no_grad():
        output = model(dense_matrix, features)
        test_loss = criterion(output[idx_test], labels.squeeze()[idx_test])
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == labels.squeeze()).sum().item()

    accuracy = correct_predictions / len(labels)

    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")

    ts = TSNE(n_components=2)
    ts.fit_transform(output[idx_test].to('cpu').detach().numpy())

    x = ts.embedding_
    y = labels[idx_test].to('cpu').detach().numpy().flatten()

    xi = []
    for i in range(output_dim):
        xi.append(x[np.where(y == i)])

    # colors = ['mediumblue', 'green', 'red', 'yellow', 'cyan', 'mediumvioletred', 'mediumspringgreen']
    # plt.figure(figsize=(8, 6))
    # for i in range(output_dim):
    #     plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, color=colors[i], marker='+', alpha=1)
    # plt.savefig("./fig/citeseer_clss")
        
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
    
    plt.savefig('./fig/citeseer_class_Pairnorm.png')  # 保存图像
    plt.show()
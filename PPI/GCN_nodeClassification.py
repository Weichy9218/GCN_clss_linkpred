import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import sys
sys.path.append("..")
sys.path.append(".")
from PPI.ppiDataset import PPIDataFromJson


class GCNLayer(nn.Module):
    def __init__(self, Input_dim, Output_dim, dropEdge_rate=0):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(Input_dim, Output_dim)
        self.dropEdge_rate = dropEdge_rate  # 存储dropEdge的概率

    def forward(self, Adjacency, Features):
        # Apply DropEdge
        if self.dropEdge_rate:
            Adjacency = self.drop_edge(Adjacency, self.dropEdge_rate)

        # 增加自环
        identity_matrix = torch.eye(Adjacency.size(0), device=Adjacency.device)
        Adjacency = Adjacency + identity_matrix

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

    @staticmethod
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


from lib import GCNConv


class GCNModel(nn.Module):
    def __init__(self, Input_dim, Hidden_dim, Output_dim):
        super(GCNModel, self).__init__()
        self.layer1 = GCNConv(Input_dim, Hidden_dim)
        self.layer2 = GCNConv(Hidden_dim, Hidden_dim)
        self.layer3 = GCNConv(Hidden_dim, Output_dim)
        self.dropout = torch.nn.Dropout(p=0.01)

    def forward(self, x, edge_index):
        # Forward pass through GCN layers
        hidden = self.layer1(x, edge_index)
        # hidden = self.PairNorm(hidden)  # pairNorm
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        hidden0 = hidden
        hidden = self.layer2(hidden, edge_index)
        hidden += hidden0  # 残差连接
        # hidden = self.PairNorm(hidden)  # pairNorm
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        Output = self.layer3(hidden, edge_index)
        # Output = self.PairNorm(Output)  # pairNorm
        Output = torch.sigmoid(Output)
        return Output


    @staticmethod
    def PairNorm(x_feature):
        col_mean = x_feature.mean(dim=0)
        x_feature = x_feature - col_mean
        row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
        x_feature = x_feature / row_norm_individual
        return x_feature


def init_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def micro_f1_score(y_pred, y_true):
    # Convert y_pred and y_true into binary tensors
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()

    tp = (y_pred * y_true).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)

    precision1 = tn / (tn + fn + 1e-16)
    recall1 = tn / (tn + fp + 1e-16)

    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    f11 = 2 * precision1 * recall1 / (precision1 + recall1 + 1e-16)

    # Calculate weights based on the number of positive and negative samples
    num_positive_samples = y_true.sum(dim=0)
    num_negative_samples = (1 - y_true).sum(dim=0)
    total_samples = num_positive_samples + num_negative_samples
    w0 = num_positive_samples / total_samples
    w1 = 1 - w0
    # Calculate weighted average of f1 score
    weighted_f1 = (f1 * w0 + f11 * w1).mean()
    # f1 =f1.mean()
    return weighted_f1


if __name__ == '__main__':

    init_seed(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_of_ppi = r"./PPI/ppi"
    ppi_dataset = PPIDataFromJson(path_of_ppi)

    num_nodes = ppi_dataset.num_nodes
    edge_index = ppi_dataset.edge_of_pg
    train_mask, val_mask, test_mask = ppi_dataset.data_partition_node()
    num_of_class = ppi_dataset.num_of_class
    feature_dim = ppi_dataset.feature_dim

    tensor_x = torch.tensor(ppi_dataset.feature_of_pg, device=device, dtype=torch.float)
    tensor_x = F.normalize(tensor_x, p=2, dim=-1)
    tensor_y = torch.tensor(ppi_dataset.label_of_pg, device=device, dtype=torch.float)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    hidden_size = 256  # 隐藏层维度，可以根据需要调整
    output_size = ppi_dataset.num_of_class  # 输出维度，例如类别数量

    model = GCNModel(feature_dim, hidden_size, output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    num_epochs = 500
    best_val_loss = float('inf')
    best_model_state_dict = None

    train_y = tensor_y[train_mask]
    tensor_adjacency = torch.tensor(edge_index, device=device, dtype=torch.long).t()

    # 初始化记录训练损失和验证准确率的列表
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(tensor_x, tensor_adjacency)
        train_mask_logits = output[train_mask]

        train_loss = criterion(train_mask_logits, train_y)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(tensor_x, tensor_adjacency)
            val_mask_logits = output[val_mask]
            val_loss = criterion(val_mask_logits, tensor_y[val_mask])

            f1_score = micro_f1_score(val_mask_logits.cpu(), tensor_y[val_mask].cpu())
            val_accuracies.append(f1_score)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, f1_score: {f1_score}")

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
        output = model(tensor_x, tensor_adjacency)
        test_mask_logits = output[test_mask]
        test_loss = criterion(test_mask_logits, tensor_y[test_mask])
        f1_score = micro_f1_score(test_mask_logits.cpu(), tensor_y[test_mask].cpu())

    print(f"Test Loss: {test_loss}, f1_score: {f1_score}")
    # 绘制训练损失和验证准确率图
    fig, ax1 = plt.subplots(figsize=(10, 8))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.plot(train_losses, label='Train Loss', color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # 实例化一个第二个y轴
    color = 'tab:blue'
    ax2.set_ylabel('Validation f1_score')
    ax2.plot(val_accuracies, label='Validation f1_score', color=color)
    ax2.tick_params(axis='y')

    # fig.tight_layout()  # 调整布局
    plt.title("Training Loss and Validation f1_score")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig('./fig/PPI_clss_nopair.png')  # 保存图像
    plt.show()
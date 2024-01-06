import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import sys
sys.path.append("..")
sys.path.append(".")
from Cora.CoraDataset import CoraDataset
import matplotlib.pyplot as plt

# 参考DL_GCN的代码
class GraphConvolution(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, use_bias=True):
        # 计算的是卷积部分D^-1/2 A D^-1/2 * X * W , X为feature，W为参数
        super(GraphConvolution, self).__init__()

        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.use_bias = use_bias

        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(in_features_dim, out_features_dim))

        # 定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight)
        # init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adj, in_feature):
        # 输入的为稀疏矩阵adj
        support = torch.mm(in_feature, self.weight)  # X*W
        Output = torch.sparse.mm(adj, support)  # A*X*W
        if self.use_bias:
            Output += self.bias  # 添加偏置项
        return Output


def get_adjacent(edge_of_pg, num_graph_node, symmetric_of_edge=True):
    """
    根据图的边信息构建邻接矩阵。
        :param edge_of_pg: (numpy.array): 图的边信息，每一行是一个边的两个节点索引。
        :param num_graph_node:(int): 图中节点的总数。
        :param symmetric_of_edge:  (bool): 是否对边进行对称处理，默认为 True。

    return:adj (scipy.sparse.coo_matrix): 构建的邻接矩阵，采用稀疏矩阵的表示方式。
    """
    # 进行对称处理
    if symmetric_of_edge:
        new_edge_of_pg = convert_symmetric(edge_of_pg)
    else:
        new_edge_of_pg = np.copy(edge_of_pg)

    # 获取边的数量和权重
    num_edges = len(new_edge_of_pg)
    graph_w = np.ones(num_edges)

    # 将边的信息转换为稀疏矩阵的 COO 表示
    np_edge = np.array(new_edge_of_pg)
    adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                        shape=[num_graph_node, num_graph_node])

    return adj


def convert_symmetric(edge_of_pg):
    """
    对图的边进行对称处理，确保图是无向图。
        :param edge_of_pg: (numpy.array): 图的边信息，每一行是一个边的两个节点索引。

    return: (numpy.array): 对称处理后的边信息。
    """
    new_edge_of_pg = []
    for edge_index in edge_of_pg:
        symmetric_edge_index = [edge_index[1], edge_index[0]]
        if symmetric_edge_index not in edge_of_pg:
            new_edge_of_pg.append(symmetric_edge_index)

    new_edge_of_pg.extend(edge_of_pg)
    return np.array(new_edge_of_pg)


def normalization(adj, self_link=True):
    """
    对邻接矩阵进行归一化操作。
    A = D^-1/2 A D^-1/2
        :param adj:(scipy.sparse.coo_matrix): 输入的邻接矩阵，采用稀疏矩阵的表示方式。
        :param self_link:(bool): 是否增加自连接，默认为 True。

    return: normalized_adj (scipy.sparse.coo_matrix): 归一化后的邻接矩阵，采用稀疏矩阵的表示方式。
    """
    adj = sp.coo_matrix(adj)  # 稀疏矩阵-->稠密矩阵
    if self_link:
        adj += sp.eye(adj.shape[0])  # 增加自连接
    row_sum = np.array(adj.sum(1))  # 对列求和，得到每一行的度
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_hat = sp.diags(d_inv_sqrt)
    normalized_adj = d_hat.dot(adj).dot(d_hat).tocoo()  # 返回coo_matrix形式
    return normalized_adj


def random_adjacent_sampler(edge_of_pg, num_graph_node, drop_edge=0.1, symmetric_of_edge=True):
    """
    通过随机隐藏部分边来构建邻接矩阵。
        :param edge_of_pg: (numpy.array): 图的边信息，每一行是一个边的两个节点索引。
        :param num_graph_node:(int): 图中节点的总数。
        :param drop_edge:(float): 随机隐藏边的概率，默认为 0.1。
        :param symmetric_of_edge:(bool): 是否对边进行对称处理，默认为 True。

    return:adj (scipy.sparse.coo_matrix): 构建的邻接矩阵，采用稀疏矩阵的表示方式。
    """
    if symmetric_of_edge:
        new_edge_of_pg = []
        edge_num = int(len(edge_of_pg))
        sampler = np.random.rand(edge_num)
        for i in range(int(edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[i])
        new_edge_of_pg = np.array(new_edge_of_pg)
        new_edge_of_pg = convert_symmetric(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    else:
        new_edge_of_pg = []
        half_edge_num = int(len(edge_of_pg) / 2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[2 * i])
                new_edge_of_pg.append(edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    return adj


def generate_tensor_adjacency_for_link(edge_index, drop_edge=0, nums_node=2708, self_link=True):
    """
    生成用于链接预测任务的稀疏张量表示的邻接矩阵。
        :param self_link: (bool): 是否增加自连接，默认为 True。
        :param edge_index:(numpy.array): 边的索引信息，每一行表示一条边的两个节点索引。
        :param drop_edge:(float): 丢弃边的概率，用于构建随机隐藏部分边的邻接矩阵，默认为 0
        :param nums_node:(int): 图中节点的总数，默认为 2708。

    return:tensor_adjacency (torch.sparse.FloatTensor): 构建的稀疏张量表示的邻接矩阵。

    Notes:
        - 如果 drop_edge 为 0，则使用原始的边信息构建邻接矩阵；否则，使用随机隐藏部分边的方式构建邻接矩阵。
        - 归一化处理采用了 normalization 函数，确保得到的邻接矩阵是归一化的。
        - 最终将邻接矩阵转化为稀疏张量表示，Tensor_adjacency 是一个 torch.sparse.FloatTensor 类型的对象。

    """
    if drop_edge == 0:
        adj = get_adjacent(edge_of_pg=edge_index, num_graph_node=nums_node)
    else:
        adj = random_adjacent_sampler(edge_of_pg=edge_index, num_graph_node=nums_node, drop_edge=drop_edge)

    normalize_adj = normalization(adj, self_link=self_link)

    # 准备将原来的coo_matrix转化到tensor形式
    index_of_coo_matrix = torch.from_numpy(np.asarray([normalize_adj.row,
                                                       normalize_adj.col]).astype('int64')).long()

    values_of_index_in_matrix = torch.from_numpy(normalize_adj.data.astype(np.float32))

    # 根据三元组构造稀疏矩阵张量,张量大小为是 (2708,2708)
    Tensor_adjacency = torch.sparse_coo_tensor(
        index_of_coo_matrix, values_of_index_in_matrix,
        torch.Size([2708, 2708]), dtype=torch.float32)
    # print(tensor_adjacency)
    return Tensor_adjacency


class LinkPredictionGCNModel(nn.Module):
    def __init__(self, Input_dim, Hidden_dim, Output_dim):
        super(LinkPredictionGCNModel, self).__init__()
        self.layer1 = GraphConvolution(Input_dim, Hidden_dim)
        self.layer2 = GraphConvolution(Hidden_dim, Hidden_dim)
        self.layer3 = GraphConvolution(Hidden_dim, Output_dim)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Adjacency, Features, pos_edge_index, neg_edge_index):
        # 通过GCN层进行前向传播  encoder部分
        hidden = self.layer1(Adjacency, Features)
        # hidden = self.PairNorm(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        # hidden = self.layer2(Adjacency, hidden)
        # hidden = self.PairNorm(hidden)
        # hidden = F.relu(hidden)
        # hidden = self.dropout(hidden)

        out_feature = self.layer3(Adjacency, hidden)

        # 合并正例和负例的边索引
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=0)

        # 通过节点对的嵌入计算相似度得分
        logits = (out_feature[edge_index[:, 0]] * out_feature[edge_index[:, 1]]).sum(dim=-1)

        return torch.sigmoid(logits)  # 在输出层应用 sigmoid 激活函数，将其映射到 (0, 1) 区间，得到最终的预测

    @staticmethod
    def PairNorm(x_feature):
        """
        每列进行了 L2 归一化
        :param x_feature:
        :return:
        """
        mode = 'PN-SI'
        scale = 1
        bias = 0
        epsilon = 1e-6
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (epsilon + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean + bias

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (epsilon + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual + bias

        if mode == 'PN-SCS':
            row_norm_individual = (epsilon + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean + bias

        return x_feature


def init_seeds(seed=1234):
    # 随机seed,保证实验可重复性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_link_labels(pos_edge_index, neg_edge_index):
    """
    获取链接预测任务的标签。
    :param pos_edge_index:(torch.Tensor): 正例边的索引，每一行表示一条正例边的两个节点索引。
    :param neg_edge_index:(torch.Tensor): 负例边的索引，每一行表示一条负例边的两个节点索引。
    :return:link_labels:(torch.Tensor): 负例边的索引，每一行表示一条负例边的两个节点索引。

     Notes:
        - num_of_edge 表示总的边的数量，link_labels 初始化为全零，然后将正例边的标签设置为 1。
        - 返回的 link_labels 是一个 torch.Tensor 对象，用于训练链接预测模型时作为标签使用。
    """
    num_of_edge = pos_edge_index.size(0) + neg_edge_index.size(0)
    link_labels = torch.zeros(num_of_edge, dtype=torch.float).to(device)
    link_labels[:pos_edge_index.size(0)] = 1.
    return link_labels


def negative_edge_sampling(train_neg_edge_index, train_pos_edge_index):
    """
    从训练集的neg边中随机采样出和pos边一样数目的边
        :param train_neg_edge_index: 负样本集合
        :param train_pos_edge_index: 正样本集合

    return:
        sampler_train_neg_edge_index:和正样本边相同数量的（维度一致）的neg边
    """
    num_pos_edge = len(train_pos_edge_index)
    num_neg_edge = len(train_neg_edge_index)
    perm = np.random.permutation(num_neg_edge)  # 随机排列
    train_neg = perm[:num_pos_edge]
    sampler_train_neg_edge_index = train_neg_edge_index[train_neg]
    return sampler_train_neg_edge_index


def generate_positive_negative_edges(Dense_matrix):
    """
    生成正负样本边
        param Dense_matrix: 稠密矩阵

    return:
    - positive_edges: 正样本边
    - negative_edges: 负样本边
    """
    nonzero_indices = np.column_stack(Dense_matrix.nonzero())  # dense.nonzero()表示找出矩阵中非零元素的索引
    sorted_data = np.sort(nonzero_indices, axis=1)  # 对列排序
    # print(sorted_data)
    Edge_of_pos = np.unique(sorted_data, axis=0)  # 只保留上上三角矩阵边
    # print(edge_of_pos.shape)
    neg_node_numThreshold = int(5 * np.sqrt(len(Edge_of_pos)))  # 定义负采样数量临界值
    edge_of_pos_pg_list = Edge_of_pos.tolist()
    Edge_of_neg = []
    # 负采样
    if dense_matrix.shape[0] <= neg_node_numThreshold:
        # 在num_graph_node*num_graph_node的上三角阵里找负边
        for i in range(dense_matrix.shape[0]):
            for j in range(i + 1, dense_matrix.shape[0]):
                edge = [i, j]
                if edge not in edge_of_pos_pg_list:
                    Edge_of_neg.append(edge)
    else:
        sampler_row = random.sample(range(0, dense_matrix.shape[0]), neg_node_numThreshold)
        for row in sampler_row:
            for col in range(row + 1, neg_node_numThreshold):
                edge = [row, col]
                if edge not in edge_of_pos_pg_list:
                    Edge_of_neg.append(edge)
    Edge_of_neg = np.array(Edge_of_neg)  # 负样本边采样
    # print(edge_of_pos, edge_of_neg)

    return Edge_of_pos, Edge_of_neg


def split_data(Edge_of_pos, Edge_of_neg):
    """
    将正负样本边集合分割为训练、验证和测试集，并对索引进行随机打乱。
        :param Edge_of_pos:正样本边集合
        :param Edge_of_neg:负样本边集合

    return:
    - edge_train_pos: 训练集中的正样本边
    - edge_val_pos: 验证集中的正样本边
    - edge_test_pos: 测试集中的正样本边
    - edge_train_neg: 训练集中的负样本边
    - edge_val_neg: 验证集中的负样本边
    - edge_test_neg: 测试集中的负样本边
   """
    # 打乱索引
    pos_node_num = len(Edge_of_pos)
    neg_node_num = len(Edge_of_neg)
    index_pos = torch.randperm(pos_node_num)
    index_neg = torch.randperm(neg_node_num)
    # 定义划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    # 计算划分点
    train_size_pos = int(train_ratio * pos_node_num)
    val_size_pos = int(val_ratio * pos_node_num)
    train_size_neg = int(train_ratio * neg_node_num)
    val_size_neg = int(val_ratio * neg_node_num)
    # 划分索引
    idx_train_pos = index_pos[:train_size_pos]
    idx_val_pos = index_pos[train_size_pos:(train_size_pos + val_size_pos)]
    idx_test_pos = index_pos[(train_size_pos + val_size_pos):]
    idx_train_neg = index_neg[:train_size_neg]
    idx_val_neg = index_neg[train_size_neg:(train_size_neg + val_size_neg)]
    idx_test_neg = index_neg[(train_size_neg + val_size_neg):]
    # print(idx_train_pos)
    # 根据索引选出样本
    Edge_of_pos_train = Edge_of_pos[idx_train_pos]
    Edge_of_pos_val = Edge_of_pos[idx_val_pos]
    Edge_of_pos_test = Edge_of_pos[idx_test_pos]
    Edge_of_neg_train = Edge_of_neg[idx_train_neg]
    Edge_of_neg_val = Edge_of_neg[idx_val_neg]
    Edge_of_neg_test = Edge_of_neg[idx_test_neg]

    return Edge_of_pos_train, Edge_of_pos_val, Edge_of_pos_test, Edge_of_neg_train, Edge_of_neg_val, Edge_of_neg_test


if __name__ == '__main__':

    init_seeds(seed=999)  # 随机seed

    # 加载数据集
    DirPath = r"./Cora/cora"
    edge_list_path = os.path.join(DirPath, 'cora.cites')
    node_features_path = os.path.join(DirPath, 'cora.content')

    cora_dataset = CoraDataset(edge_list_path, node_features_path)
    features, labels, dense_matrix = cora_dataset.load_data(edge_list_path, node_features_path)  # 加载数据

    edge_of_pos, edge_of_neg = generate_positive_negative_edges(dense_matrix)  # 生成正负样本边

    edge_of_pos_train, edge_of_pos_val, edge_of_pos_test, edge_of_neg_train, edge_of_neg_val, edge_of_neg_test = split_data(
        edge_of_pos, edge_of_neg)  # 数据划分

    edge_of_neg_train = negative_edge_sampling(edge_of_neg_train, edge_of_pos_train)  # 采样和正样本一样多的负样本
    edge_of_neg_val = negative_edge_sampling(edge_of_neg_val, edge_of_pos_val)
    edge_of_neg_test = negative_edge_sampling(edge_of_neg_test, edge_of_pos_test)
    # print(edge_of_pos_train)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模型、优化器和损失函数
    input_size = 1433
    hidden_size = 128
    output_size = 2
    num_epochs = 100
    selfloop = True
    learning_rate = 1e-3

    model = LinkPredictionGCNModel(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # 将特征、邻接矩阵和标签移到设备
    features = torch.tensor(features, dtype=torch.float32).to(device)
    features = F.normalize(features, p=2, dim=-1)  # 特征列归一化
    tensor_adjacency = generate_tensor_adjacency_for_link(edge_of_pos_train,
        drop_edge=0, nums_node=len(labels), self_link=selfloop).to(device)  # 链接预测任务的稀疏张量表示的邻接矩阵
    edge_of_pos_train = torch.tensor(edge_of_pos_train, dtype=torch.long).to(device)
    edge_of_neg_train = torch.tensor(edge_of_neg_train, dtype=torch.long).to(device)
    edge_of_pos_val = torch.tensor(edge_of_pos_val, dtype=torch.long).to(device)
    edge_of_neg_val = torch.tensor(edge_of_neg_val, dtype=torch.long).to(device)
    edge_of_pos_test = torch.tensor(edge_of_pos_test, dtype=torch.long).to(device)
    edge_of_neg_test = torch.tensor(edge_of_neg_test, dtype=torch.long).to(device)

    # 训练
    best_val_loss = float('inf')
    best_model_state_dict = None

    # 初始化记录训练损失和验证准确率的列表
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_output = model(tensor_adjacency, features, edge_of_pos_train, edge_of_neg_train)
        targets = get_link_labels(edge_of_pos_train, edge_of_neg_train)

        # 计算二元交叉熵损失
        train_loss = criterion(train_output, targets)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(tensor_adjacency, features, edge_of_pos_val, edge_of_neg_val)
            targets = get_link_labels(edge_of_pos_val, edge_of_neg_val)
            val_loss = criterion(val_output, targets)
            auc_roc = roc_auc_score(targets.cpu().numpy().flatten(),
                                    val_output.cpu().numpy().flatten()
                                    )
            val_accuracies.append(auc_roc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        # 保存最佳模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()

    # 选择最佳模型进行测试
    model.load_state_dict(best_model_state_dict)

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        test_output = model(tensor_adjacency, features, edge_of_pos_test, edge_of_neg_test)
        targets = get_link_labels(edge_of_pos_test, edge_of_neg_test)
        test_loss = criterion(test_output, targets)

        auc_roc = roc_auc_score(targets.cpu().numpy().flatten(),
                                test_output.cpu().numpy().flatten()
                                )
        print(f"Test Loss: {test_loss},Test AUC-ROC: {auc_roc}")

    # 绘制训练损失和验证准确率图
    fig, ax1 = plt.subplots(figsize=(10, 8))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.plot(train_losses, label='Train Loss', color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # 实例化一个第二个y轴
    color = 'tab:blue'
    ax2.set_ylabel('Validation AUC')
    ax2.plot(val_accuracies, label='Validation AUC', color=color)
    ax2.tick_params(axis='y')

    # fig.tight_layout()  # 调整布局
    plt.title("Training Loss and Validation AUC")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig('./fig/cora_link.png')  # 保存图像
    plt.show()
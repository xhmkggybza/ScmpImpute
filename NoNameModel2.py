import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix, find
from sklearn.utils.extmath import randomized_svd

def compute_markov(data, k=10, epsilon=1, distance_metric='euclidean', ka=0):

    SCD = data.clone().detach().cpu().numpy()  # torch没有自带的最近邻搜索算法，不想自己写，这里先将张量复制，剥离出计算图，然后转化为numpy

    N = SCD.shape[0]  # 获取细胞数量

    # Nearest neighbors
    print('Computing distances')
    # 近邻搜索算法
    nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(SCD)
    # distance 数据点和最近邻的距离
    # indices 数据点的最近邻的index
    distances, indices = nbrs.kneighbors(SCD)

    if ka > 0:
        print('Autotuning distances')
        # 生成一个从N-1到0的反向迭代器
        for j in reversed(range(N)):
            temp = sorted(distances[j])
            lMaxTempIdxs = min(ka, len(temp))
            if lMaxTempIdxs == 0 or temp[lMaxTempIdxs] == 0:
                distances[j] = 0
            else:
                distances[j] = np.divide(distances[j], temp[lMaxTempIdxs])

    # Adjacency matrix
    print('Computing kernel')
    rows = np.zeros(N * k, dtype=np.int32)  # 这他妈是一个一维的数据，草
    cols = np.zeros(N * k, dtype=np.int32)
    dists = np.zeros(N * k)
    location = 0
    for i in range(N):
        inds = range(location, location + k)
        rows[inds] = indices[i, :] # 这里对应的是前几个邻居的索引
        cols[inds] = i
        dists[inds] = distances[i, :]  # 和前几个邻居的距离
        location += k
    if epsilon > 0:
        W = csr_matrix( (dists, (rows, cols)), shape=[N, N] )
    else:
        W = csr_matrix( (np.ones(dists.shape), (rows, cols)), shape=[N, N] )

    # Symmetrize W 对称化
    W = W + W.T

    if epsilon > 0:
        # Convert to affinity (with selfloops)
        rows, cols, dists = find(W)
        rows = np.append(rows, range(N))
        cols = np.append(cols, range(N))
        dists = np.append(dists/(epsilon ** 2), np.zeros(N))
        W = csr_matrix( (np.exp(-dists), (rows, cols)), shape=[N, N] )

    # Create D
    '''D = np.ravel(W.sum(axis = 1))
    D[D!=0] = 1/D[D!=0]

    #markov normalization
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)'''

    T = W.toarray()
    T = torch.from_numpy(T).to(torch.float32)
    T = T.to('cuda')

    return T


def compute_markov_tensor(query, key, knn = False, k = 0, epsilon=1, ka = 0):
    '''
    :param query: Q矩阵
    :param key: K矩阵
    :param knn: 是否采用knn减枝 default: False
    :param k: knn的参数
    :param epsilon: 计算精度
    :param ka: 自适应核参数
    :return:
    '''
    N1 = len(query)
    N2 = len(key)
    A = torch.empty(N1, N2, device='cuda')
    for i in range(N1):
        distances = torch.sqrt(torch.sum((key - query[i]) ** 2, dim=1)+ 1e-10)

        # 自适应k的大小
        if N1 < 10:
            k = N1
            knn = False
        if 10 <= N1 < 15:
            k = 10
        if 15 <= N1 < 300:
            k = 15
        if N1 >= 300:
            k = 30

        ka = int(k / 3)

        if ka > 0:
            temp, _ = torch.sort(distances)
            lMaxTempIdxs = min(ka, len(temp))
            distances = distances.div(temp[lMaxTempIdxs]+ 1e-10) # 将 distances[j] 中的每个元素除以 temp[lMaxTempIdxs] 的值，并将结果重新存储回 distances[j] 中。

        # Convert to affinity
        distances = (distances.div(epsilon**2))**2
        aff = torch.exp(-distances)


        # 剪枝操作
        if knn:
            order = torch.argsort(aff,descending=True)
            aff[order[k:]] = 0.0

        # with selfloops
        aff[i] = 1.0

        A[i] = aff

    # 对称化处理
    A = A + A.T

    return A

# 上述函数的优化版本
def compute_markov_tensor_optic(query, key, knn=True, k=0, epsilon=1, ka=0):
    '''
    :param query: Q矩阵
    :param key: K矩阵
    :param knn: 是否采用knn减枝 default: False
    :param k: knn的参数
    :param epsilon: 计算精度
    :param ka: 自适应核参数
    :return: Markov 核矩阵
    '''
    device = query.device  # 使用 query 张量的设备
    N1 = query.size(0)
    N2 = key.size(0)

    # 计算所有 pairwise distances
    query_expanded = query.unsqueeze(1)  # Shape: (N1, 1, D)
    key_expanded = key.unsqueeze(0)  # Shape: (1, N2, D)

    distances = torch.sqrt(torch.sum((query_expanded - key_expanded) ** 2, dim=2)+ 1e-10)  # Shape: (N1, N2)

    # 根据细胞数量自适应k的大小
    if N1 < 10:
        k = N1
        knn = False
    if 10 <= N1 < 15:
        k = 10
    if 15 <= N1 < 300:
        k = 15
    if N1 >= 300:
        k = 30

    ka = int(k/3)

    # 自适应核大小
    if ka > 0:
        # 找到每行的前 ka 个最小值的索引
        _, sorted_indices = torch.sort(distances, dim=1)
        max_distances = distances[torch.arange(N1, device=device), sorted_indices[:, ka]]
        distances = distances / (max_distances.unsqueeze(1) + 1e-10)  # 防止除以零
    # Convert to affinity
    aff = torch.exp(- (distances / epsilon ** 2) ** 2)  # Shape: (N1, N2)



    if knn:
        # KNN剪枝操作
        topk_values, _ = torch.topk(aff, k, dim=1, largest=True, sorted=False)
        mask = aff >= topk_values[:, -1].unsqueeze(1)
        aff = aff * mask.float()




    # with selfloops
    #aff.fill_diagonal_(1.0)
    diag_indices = torch.arange(aff.size(0), device=aff.device)  # 获取对角线的索引
    aff[diag_indices, diag_indices] = 1.0  # 填充对角线为 1.0

    # 对称化处理
    aff = aff + aff.T


    return aff




#  基于细胞细胞相似性的模块：模型代码部分
class Cell2Cell(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, out_dropout_prob=0.2):
        super(Cell2Cell, self).__init__()

        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size


        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        #self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            Wq = nn.Linear(input_size, hidden_size)
            Wk = nn.Linear(input_size, hidden_size)
            self.query_head.append(Wq)
            self.key_head.append(Wk)
            #self.value_head.append(Wv)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        # self.dense = nn.Linear(hidden_size, input_size)
        # self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(out_dropout_prob)


    def forward(self, input_tensor, return_attention=False, att_drop=False, out_drop = False, t = 6):

        # [cell, emb] <- [cell, genes], 细胞基因表达谱投影到低维空间
        #input_tensor = self.Encoder(input_tensor)
        #
        outputs = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            #value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            #value_layer = value(input_tensor)

            # 通过马尔可夫亲和度的方式计算注意力分数
            attention_scores = compute_markov_tensor_optic(query_layer, key_layer)


            # [cells, cells] = [cell, emb]*[emb, cell]
            #attention_scores = torch.matmul(query_layer, key_layer.transpose(
            #    -1, -2))

            #attention_scores = attention_scores / math.sqrt(
            #    self.hidden_size)

            #fill_diagonal_(attention_scores, 1)


            if return_attention:
                return attention_scores

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if att_drop:
                attention_probs = self.attn_dropout(attention_probs)

            # t次扩散
            attention_probs_t = torch.linalg.matrix_power(attention_probs, t)
            # [cells, emb] = [cells, cells] * [cells, emb]
            context_layer = torch.matmul(attention_probs_t, input_tensor)

            outputs.append(context_layer)
        # avg([heads, cells, emb])
        # torch.stack 可以将由张量组成的列表，生成一个新的张量，这里是三维的
        # torch.mean 沿指定维度计算均值，这里是从第一个维度，所以会从3维变成2维
        output = torch.mean(torch.stack(outputs), 0)

        #hidden_states = self.dense(output)
        if out_drop:
            output = self.out_dropout(output)
        '''hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)'''

        return output

class Gene2Gene(nn.Module):
    def __init__(self, input_size, encoder_layer_size, n_clusters, dropout_rate):
        super(Gene2Gene, self).__init__()

        len_layer = len(encoder_layer_size)


        encoder = []
        current_dim = input_size
        for i in range(len_layer):
            l_size = encoder_layer_size[i]
            encoder.append(nn.Linear(current_dim, l_size))
            encoder.append(nn.ReLU())
            current_dim = l_size
        self.encoder = nn.Sequential(*encoder)

        self.Endropout = nn.Dropout(dropout_rate)

        decoder = []
        for i in range(len_layer-1):
            l_size = encoder_layer_size[-2 - i]
            decoder.append(nn.Linear(current_dim, l_size))
            decoder.append(nn.ReLU())
            current_dim = l_size
        decoder.append(nn.Linear(current_dim, input_size))
        decoder.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder)

        self.classifier_layer = nn.Linear(encoder_layer_size[-1], n_clusters)

    def forward(self, input_tensor,is_drop=False):

        # 编码器压缩
        hidden_tensor = self.encoder(input_tensor)

        if is_drop:
            hidden_tensor = self.Endropout(hidden_tensor)

        recon_tensor = self.decoder(hidden_tensor)

        classfication_result = self.classifier_layer(hidden_tensor)
        classfication_result = nn.Softmax(dim=-1)(classfication_result)

        return recon_tensor, classfication_result

#  基于细胞细胞相似性的模块：模型代码部分
class Cell2CellwithAuto(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, n_clusters, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, out_dropout_prob=0.2):
        super(Cell2CellwithAuto, self).__init__()

        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size


        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            Wqk = nn.Linear(input_size, hidden_size)
            #Wk = nn.Linear(input_size, hidden_size)
            Wv = nn.Linear(input_size, hidden_size)
            self.query_head.append(Wqk)
            self.key_head.append(Wqk)
            self.value_head.append(Wv)

        self.relu = nn.LeakyReLU()
        self.Endropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.softplus = nn.Softplus()

        self.classifier_layer = nn.Linear(hidden_size, n_clusters)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        # self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        #self.out_dropout = nn.Dropout(out_dropout_prob)


    def forward(self, input_tensor, return_attention=False, attention_drop=False, t = 6, hidden_drop = False):

        hidden = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            # 通过马尔可夫亲和度的方式计算注意力分数
            attention_scores = compute_markov_tensor_optic(query_layer, key_layer)


            # [cells, cells] = [cell, emb]*[emb, cell]
            #attention_scores = torch.matmul(query_layer, key_layer.transpose(
            #    -1, -2))

            #attention_scores = attention_scores / math.sqrt(
            #    self.hidden_size)

            #fill_diagonal_(attention_scores, 1)


            if return_attention:
                return attention_scores

            # Normalize the attention scores to probabilities.
            attention_sums = attention_scores.sum(dim = 1,keepdim=True)
            attention_probs = attention_scores / attention_sums
            #attention_probs = nn.Softmax(dim=-1)(attention_scores)   不采用softmax，因为这是指数归一化，剪枝之后零值太多。

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if attention_drop:
                attention_probs = self.attn_dropout(attention_probs)

            # t次扩散
            attention_probs_t = torch.linalg.matrix_power(attention_probs, t)
            # [cells, emb] = [cells, cells] * [cells, emb]
            context_layer = torch.matmul(attention_probs_t, value_layer)


            hidden.append(context_layer)
        # avg([heads, cells, emb])
        # torch.stack 可以将由张量组成的列表，生成一个新的张量，这里是三维的
        # torch.mean 沿指定维度计算均值，这里是从第一个维度，所以会从3维变成2维
        hidden = torch.mean(torch.stack(hidden), 0)
        hidden = self.relu(hidden)

        if hidden_drop:
            hidden = self.Endropout(hidden)

        # 下面是两个方向，一个用来解码，一个用来分类
        recon_tensor = self.softplus(self.decoder(hidden))

        classfication_result = self.classifier_layer(hidden)
        classfication_result = nn.Softmax(dim=-1)(classfication_result)

        return recon_tensor, classfication_result

# 使用python实现矩阵的自适应阈值低秩逼近
def lowRank_Approximation(data, k=0, q=10, quantile_prob = 0.001):
    '''
    :param data: 表达矩阵，行为细胞，列为基因
    :param k: 矩阵补全要逼近的秩
    :param q: 随机化奇异值分解的附加幂迭代次数，用于提高计算结果的精度。
    :return:
    '''
    if k == 0:
        k, _, _ = choose_k(data)
        print("估计秩为：", k)

    # 获得非零值信息，一个bool矩阵，原表达矩阵不为0的话对应的值为True
    originally_nonzero = data > 0

    # 开始补全
    U, S, Vt = randomized_svd(data, n_components=k, n_iter=q, random_state=42)
    data_rank_k = np.dot(U, np.dot(np.diag(S), Vt))

    # 计算每一列的基因的分位数
    data_rank_k_mins = np.abs(np.quantile(data_rank_k, quantile_prob, axis=0))
    # 将矩阵中每一列中小于对应分位数的元素置为0
    data_rank_k_cor = np.copy(data_rank_k)
    data_rank_k_cor[data_rank_k_cor <= data_rank_k_mins] = 0

    # 计算每一列非零元素的标准差
    sigma_1 = np.apply_along_axis(sd_nonzero, axis=0, arr=data_rank_k_cor)
    sigma_2 = np.apply_along_axis(sd_nonzero, axis=0, arr=data)

    # 计算每列非零元素的均值
    mu_1 = np.apply_along_axis(mean_nonzero, axis=0, arr=data_rank_k_cor)
    mu_2 = np.apply_along_axis(mean_nonzero, axis=0, arr=data)

    # toscale <- !is.na(sigma_1) & !is.na(sigma_2) & !(sigma_1 == 0 & sigma_2 == 0) & !(sigma_1 == 0)  确定要进行缩放的列索引-
    toscale = (~np.isnan(sigma_1)) & (~np.isnan(sigma_2)) & (~((sigma_1 == 0) & (sigma_2 == 0))) & (~(sigma_1 == 0))

    sigma_1_2 = sigma_2 / sigma_1
    toadd = -1*mu_1*sigma_2/sigma_1 + mu_2

    data_rank_k_temp = data_rank_k_cor[:,toscale] # 选择可以缩放的列
    data_rank_k_temp = data_rank_k_temp * sigma_1_2[toscale]
    data_rank_k_temp = data_rank_k_temp + toadd[toscale]

    data_rank_k_cor_sc = data_rank_k_cor
    data_rank_k_cor_sc[:,toscale] = data_rank_k_temp
    data_rank_k_cor_sc[data_rank_k_cor == 0] = 0

    lt0 = data_rank_k_cor_sc < 0
    data_rank_k_cor_sc[lt0] = 0

    # 把原始矩阵中不为0，在经过填补缩放一系列操作之后变0的元素恢复
    # A_norm_rank_k_cor_sc[originally_nonzero & A_norm_rank_k_cor_sc ==0] <- A_norm[originally_nonzero & A_norm_rank_k_cor_sc ==0]
    data_rank_k_cor_sc[originally_nonzero & (data_rank_k_cor_sc == 0)] = data[originally_nonzero & (data_rank_k_cor_sc == 0)]
    return data_rank_k_cor_sc


def choose_k(data, K = 100, thresh = 6, noise_start = 79, q = 2):
    '''
    为低秩矩阵补全选择合适的阈值
    :param data: 表达矩阵，行为细胞，列为基因
    :param K: 保留的奇异值和奇异向量的数量
    :param thresh: 筛选K的阈值
    :param noise_start:
    :param q: 随机化奇异值分解的附加幂迭代次数，用于提高计算结果的精度。
    :return:
    '''

    nrow = data.shape[0]
    ncol = data.shape[1]

    if K > min(nrow, ncol):
        print("For an m by n matrix, K must be smaller than the min(m,n).\n")
        return

    if noise_start > K-5:
        print("There need to be at least 5 singular values considered noise.\n")


    # 随机化矩阵奇异值分解
    # U 奇异向量，以列为单位
    # S 奇异值，并且是降序排序的，一维数组的形式返回
    # Vt 奇异向量， 以行为单位
    U, S, Vt = randomized_svd(data, n_components=K, n_iter=q, random_state=42)

    diffs = S[0:len(S)-1] - S[1:len(S)]
    mu = np.mean(diffs[noise_start-1:len(diffs)])
    sigma = np.std(diffs[noise_start-1:len(diffs)], ddof=1)
    num_of_sds = (diffs - mu) / sigma

    # 解释下这里为什么要加1，因为tmd比python从0开始，这里我们要返回的是选择到的秩（也就是奇异值的数量）
    # 比如这里满足条件的是 第50个元素，也就是经过处理的第50个奇异值和第51个奇异值之间的距离满足条件，再往后距离就不满足了
    # 所以这里的低秩逼近只需要前50个奇异值，也就是0-50，也就是51个奇异值，所以返回值要加1
    k = np.max(np.where(num_of_sds > thresh)[0])+1

    return k, num_of_sds, S


# 定义一个函数来计算每列非零元素的标准差
def sd_nonzero(x):
    non_zero_elements = x[x != 0]  # 过滤掉零元素
    return np.std(non_zero_elements, ddof=1) if len(non_zero_elements) > 0 else np.nan

# 计算每列非零元素的均值
def mean_nonzero(x):
    non_zero_elements = x[x != 0]  # 过滤掉零元素
    return np.mean(non_zero_elements) if len(non_zero_elements) > 0 else np.nan


def Cell_Specific_Weights(X_recon_list, Xraw, k = 21, distance_metric='euclidean'):
    """
    计算细胞特异性结构权重
    :param X_recon_list: 包含2-3个途径获得的补全后的基因表达数据
    :param Xraw: 原始基因表达数据
    :param k: 最邻近数量, Default:31, 为什么要多个1，是因为自环问题，要去掉第一个邻居（也就是样本自身）
    :return: 最终插补基因表达数据
    """
    nCells = Xraw.shape[0]
    Xp_list = []
    Xfn_list = []
    Xkn_list = []
    ka = int(k/3)
    for X in X_recon_list:
        # 使用最近邻搜索
        nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(X)
        # _ 数据点和最近邻的距离
        # indices 数据点的最近邻的index
        _, indices = nbrs.kneighbors(X)

        # 由于不要自环，这里索引把第一项删掉
        indices = indices[:, 1:]  # [nCells, k-1]

        # 根据knn邻居对原表达谱的预测
        X_neighber = X[indices] # 这里会得到一个三维的数据，其中每一个矩阵代表对应的相应邻居的表达谱 [nCells, k-1, nGenes]
        Xp = np.mean(X_neighber, axis=1) # 计算每一个细胞的邻域细胞表达谱均值 [nCells, nGenes]
        Xp_list.append(Xp)

        # 拿到每一个细胞的最近的邻居的插值表达谱
        Xfn = X[indices[:, 0]]  # [nCells, nGenes]
        Xfn_list.append(Xfn)

        # 拿到每一个细胞的第ka个邻居的插值表达谱
        Xkn = X[indices[:, ka]]  # [nCells, nGenes]
        Xkn_list.append(Xkn)

    # 计算亲和度
    Aff = np.zeros((len(Xp_list), nCells))
    for i in range(len(Xp_list)):
        Xp = Xp_list[i]  # [nCells, nGenes]
        Xfn = Xfn_list[i]
        Xkn = Xkn_list[i]  # [nCells, nGenes]

        # 计算原始表达数据和插值邻域预测之间的欧式距离
        Diff = Xraw - Xp  # [nCells, nGenes]
        Distance = np.sqrt(np.sum(Diff**2, axis=1))  # [nCells,]

        # 计算原始表达谱和插值表达的第一个邻居的欧式距离
        #Diff_fn = Xraw - Xfn
        #Distance_fn = np.sqrt(np.sum(Diff_fn**2, axis=1))

        # 计算原始表达谱和插值表达的第ka邻居的欧式距离
        Diff_kn = Xraw - Xkn
        Distance_kn = np.sqrt(np.sum(Diff_kn**2, axis=1))

        # 原始表达数据和插值邻域预测之间的欧式距离 减去 原始表达数据和插值表达的第一个邻居的欧式距离， 防止局部连通性
        #DistanceWithout_fn = Distance-Distance_fn
        #DistanceWithout_fn[DistanceWithout_fn < 0] = 0

        A = np.exp(-1*Distance/(Distance_kn+1e-10))  # [nCells,]
        Aff[i] = A

    # 将亲和读转化为细胞特异性权重
    W= softmax(Aff)  # [N, nCells]
    Xrecon = np.zeros(Xraw.shape)
    for i in range(len(W)):
        Ws = W[i]  # [nCells,]
        Xrecon = Xrecon + X_recon_list[i]*Ws[:,np.newaxis]


    return Xrecon



def softmax(x):
    # x 是二维数组
    # 每一列的最大值，用于数值稳定性，防止溢出
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)























import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix, find
from sklearn.utils.extmath import randomized_svd


def compute_markov_tensor_optic(query, key, knn=True, k=0, epsilon=1, ka=0):
    '''
    :param query: Q matrix
    :param key: K matrix
    :param knn: Should KNN pruning be used, default: False
    :param k: knn parameters
    :param epsilon: Calculation accuracy
    :param ka: Adaptive kernel parameters
    :return: Markov kernel matrix
    '''
    device = query.device
    N1 = query.size(0)
    N2 = key.size(0)

    # pairwise distances
    query_expanded = query.unsqueeze(1)  # Shape: (N1, 1, D)
    key_expanded = key.unsqueeze(0)  # Shape: (1, N2, D)

    distances = torch.sqrt(torch.sum((query_expanded - key_expanded) ** 2, dim=2)+ 1e-10)  # Shape: (N1, N2)

    # The size of k adapts to the cell number
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

    if ka > 0:
        _, sorted_indices = torch.sort(distances, dim=1)
        max_distances = distances[torch.arange(N1, device=device), sorted_indices[:, ka]]
        distances = distances / (max_distances.unsqueeze(1) + 1e-10)  
    # Convert to affinity
    aff = torch.exp(- (distances / epsilon ** 2) ** 2)  # Shape: (N1, N2)


    if knn:
        topk_values, _ = torch.topk(aff, k, dim=1, largest=True, sorted=False)
        mask = aff >= topk_values[:, -1].unsqueeze(1)
        aff = aff * mask.float()




    # with selfloops
    #aff.fill_diagonal_(1.0)
    diag_indices = torch.arange(aff.size(0), device=aff.device)  
    aff[diag_indices, diag_indices] = 1.0  

    aff = aff + aff.T


    return aff

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



    def forward(self, input_tensor, return_attention=False, attention_drop=False, t = 6, hidden_drop = False):

        hidden = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            attention_scores = compute_markov_tensor_optic(query_layer, key_layer)
            if return_attention:
                return attention_scores

            # Normalize the attention scores to probabilities.
            attention_sums = attention_scores.sum(dim = 1,keepdim=True)
            attention_probs = attention_scores / attention_sums

            if attention_drop:
                attention_probs = self.attn_dropout(attention_probs)

            attention_probs_t = torch.linalg.matrix_power(attention_probs, t)
            # [cells, emb] = [cells, cells] * [cells, emb]
            context_layer = torch.matmul(attention_probs_t, value_layer)


            hidden.append(context_layer)
            
        hidden = torch.mean(torch.stack(hidden), 0)
        hidden = self.relu(hidden)

        if hidden_drop:
            hidden = self.Endropout(hidden)

        recon_tensor = self.softplus(self.decoder(hidden))

        classfication_result = self.classifier_layer(hidden)
        classfication_result = nn.Softmax(dim=-1)(classfication_result)

        return recon_tensor, classfication_result


def lowRank_Approximation(data, k=0, q=10, quantile_prob = 0.001):
    '''
    :param data: Expression matrix, rows represent cells, columns represent genes
    :param k: Matrix completion aims to approximate the rank.
    :param q: Randomized singular value decomposition is used to increase the number of additional exponential iterations to improve the accuracy of the computation results.
    :return:
    '''
    if k == 0:
        k, _, _ = choose_k(data)
        print("估计秩为：", k)

    originally_nonzero = data > 0

    U, S, Vt = randomized_svd(data, n_components=k, n_iter=q, random_state=42)
    data_rank_k = np.dot(U, np.dot(np.diag(S), Vt))

    data_rank_k_mins = np.abs(np.quantile(data_rank_k, quantile_prob, axis=0))
    data_rank_k_cor = np.copy(data_rank_k)
    data_rank_k_cor[data_rank_k_cor <= data_rank_k_mins] = 0

    sigma_1 = np.apply_along_axis(sd_nonzero, axis=0, arr=data_rank_k_cor)
    sigma_2 = np.apply_along_axis(sd_nonzero, axis=0, arr=data)

    mu_1 = np.apply_along_axis(mean_nonzero, axis=0, arr=data_rank_k_cor)
    mu_2 = np.apply_along_axis(mean_nonzero, axis=0, arr=data)

    # toscale <- !is.na(sigma_1) & !is.na(sigma_2) & !(sigma_1 == 0 & sigma_2 == 0) & !(sigma_1 == 0)  
    toscale = (~np.isnan(sigma_1)) & (~np.isnan(sigma_2)) & (~((sigma_1 == 0) & (sigma_2 == 0))) & (~(sigma_1 == 0))

    sigma_1_2 = sigma_2 / sigma_1
    toadd = -1*mu_1*sigma_2/sigma_1 + mu_2

    data_rank_k_temp = data_rank_k_cor[:,toscale] 
    data_rank_k_temp = data_rank_k_temp * sigma_1_2[toscale]
    data_rank_k_temp = data_rank_k_temp + toadd[toscale]

    data_rank_k_cor_sc = data_rank_k_cor
    data_rank_k_cor_sc[:,toscale] = data_rank_k_temp
    data_rank_k_cor_sc[data_rank_k_cor == 0] = 0

    lt0 = data_rank_k_cor_sc < 0
    data_rank_k_cor_sc[lt0] = 0

    # A_norm_rank_k_cor_sc[originally_nonzero & A_norm_rank_k_cor_sc ==0] <- A_norm[originally_nonzero & A_norm_rank_k_cor_sc ==0]
    data_rank_k_cor_sc[originally_nonzero & (data_rank_k_cor_sc == 0)] = data[originally_nonzero & (data_rank_k_cor_sc == 0)]
    return data_rank_k_cor_sc


def choose_k(data, K = 100, thresh = 6, noise_start = 79, q = 2):
    '''
    Choosing an appropriate threshold for low-rank matrix completion
    :param data: Expression matrix, rows represent cells, columns represent genes
    :param K: The number of singular values ​​and singular vectors retained
    :param thresh: Threshold for filtering K
    :param q: Randomized singular value decomposition is used to increase the number of additional exponential iterations to improve the accuracy of the computation results.
    :return:
    '''

    nrow = data.shape[0]
    ncol = data.shape[1]

    if K > min(nrow, ncol):
        print("For an m by n matrix, K must be smaller than the min(m,n).\n")
        return

    if noise_start > K-5:
        print("There need to be at least 5 singular values considered noise.\n")

    U, S, Vt = randomized_svd(data, n_components=K, n_iter=q, random_state=42)

    diffs = S[0:len(S)-1] - S[1:len(S)]
    mu = np.mean(diffs[noise_start-1:len(diffs)])
    sigma = np.std(diffs[noise_start-1:len(diffs)], ddof=1)
    num_of_sds = (diffs - mu) / sigma

    k = np.max(np.where(num_of_sds > thresh)[0])+1

    return k, num_of_sds, S



def sd_nonzero(x):
    non_zero_elements = x[x != 0]  
    return np.std(non_zero_elements, ddof=1) if len(non_zero_elements) > 0 else np.nan

def mean_nonzero(x):
    non_zero_elements = x[x != 0]  
    return np.mean(non_zero_elements) if len(non_zero_elements) > 0 else np.nan


def Cell_Specific_Weights(X_recon_list, Xraw, k = 21, distance_metric='euclidean'):
    """
    Calculate cell-specific structure weights
    :param X_recon_list: Includes complete gene expression data obtained through two pathways.
    :param Xraw: Original gene expression data
    :param k: The nearest neighbor count
    """
    nCells = Xraw.shape[0]
    Xp_list = []
    Xfn_list = []
    Xkn_list = []
    ka = int(k/3)
    for X in X_recon_list:

        nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(X)

        _, indices = nbrs.kneighbors(X)


        indices = indices[:, 1:]  # [nCells, k-1]


        X_neighber = X[indices] 
        Xp = np.mean(X_neighber, axis=1) 
        Xp_list.append(Xp)


        Xfn = X[indices[:, 0]]  # [nCells, nGenes]
        Xfn_list.append(Xfn)


        Xkn = X[indices[:, ka]]  # [nCells, nGenes]
        Xkn_list.append(Xkn)


    Aff = np.zeros((len(Xp_list), nCells))
    for i in range(len(Xp_list)):
        Xp = Xp_list[i]  # [nCells, nGenes]
        Xfn = Xfn_list[i]
        Xkn = Xkn_list[i]  # [nCells, nGenes]


        Diff = Xraw - Xp  # [nCells, nGenes]
        Distance = np.sqrt(np.sum(Diff**2, axis=1))  # [nCells,]


        #Diff_fn = Xraw - Xfn
        #Distance_fn = np.sqrt(np.sum(Diff_fn**2, axis=1))


        Diff_kn = Xraw - Xkn
        Distance_kn = np.sqrt(np.sum(Diff_kn**2, axis=1))


        #DistanceWithout_fn = Distance-Distance_fn
        #DistanceWithout_fn[DistanceWithout_fn < 0] = 0

        A = np.exp(-1*Distance/(Distance_kn+1e-10))  # [nCells,]
        Aff[i] = A


    W= softmax(Aff)  # [N, nCells]
    Xrecon = np.zeros(Xraw.shape)
    for i in range(len(W)):
        Ws = W[i]  # [nCells,]
        Xrecon = Xrecon + X_recon_list[i]*Ws[:,np.newaxis]


    return Xrecon



def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)























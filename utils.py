import numpy as np
import torch
import random
import sys
import os
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from igraph import *
import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from graph_function import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
import warnings
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score



def set_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def loadscExpression(csvFilename, sparseMode=True):
    '''
    Load sc Expression: rows are genes, cols are cells, first col is the gene name, first row is the cell name.
    sparseMode for loading huge datasets in sparse coding
    '''
    if sparseMode:
        print('Load expression matrix in sparseMode')
        genelist = []
        celllist = []
        with open(csvFilename.replace('.csv', '_sparse.npy'), 'rb') as f:
            objects = pkl.load(f, encoding='latin1')
        matrix = objects.tolil()

        with open(csvFilename.replace('.csv', '_gene.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                genelist.append(line)

        with open(csvFilename.replace('.csv', '_cell.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                celllist.append(line)

    else:
        print('Load expression in csv format')
        matrix = pd.read_csv(csvFilename, index_col=0)
        celllist = matrix.index.tolist()
        genelist = matrix.columns.values.tolist()
        matrix = matrix.to_numpy()
        matrix = matrix.astype(np.float32)

    return matrix, genelist, celllist


def mask(data_train, masked_prob,seed):
    index_pair_train = np.where(data_train != 0)
    np.random.seed(seed)
    masking_idx_train = np.random.choice(index_pair_train[0].shape[0], int(index_pair_train[0].shape[0] * masked_prob), replace = False)
    #to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
    X_train = copy.deepcopy(data_train)   
    X_train[index_pair_train[0][masking_idx_train], index_pair_train[1][masking_idx_train]] = 0
    return X_train, index_pair_train, masking_idx_train


def cluster(data, n_pca_components, byVar = False, random_pca=True, graphType = 'KNNgraph', para = None, adjTag = True, resolution = 'auto', NeedLouvain = True, k = 9):

    print('doing PCA')
    if byVar:
        n = np.min((data.shape[0], data.shape[1]))
        solver = 'randomized'
        if random_pca != True:
            solver = 'full'
        pca = PCA(n_components=n, svd_solver=solver)
        pcs = pca.fit_transform(data)
        var = (pca.explained_variance_ratio_).cumsum()
        npc_raw = (np.where(var > 0.7))[0].min()  # number of PC used in K-means
        if npc_raw > n_pca_components:
            npc_raw = n_pca_components
        pca_projected_data = pcs[:, :npc_raw]
    else:
        pca_projected_data = run_pca(data, n_components=n_pca_components, random=random_pca)


    if resolution == 'auto':
        if data.shape[0] < 2000:
            resolution = 0.8
        else:
            resolution = 0.5
    else:
        resolution = float(resolution)

    # K-means was used for pre-clustering, and the number of clusters was determined by the Louvain algorithm.
    adj, edgeList = generateAdj(pca_projected_data, graphType = graphType, para = para, adjTag = adjTag)
    if NeedLouvain:
        listResult, size = generateLouvainCluster(edgeList, data.shape[0])
        k = len(np.unique(listResult))
        print('Louvain cluster: ' + str(k))
        k = int(k * resolution) if int(k * resolution) >= 3 else 2

    clustering = KMeans(n_clusters=k, random_state=0).fit(pca_projected_data)
    listResult = clustering.labels_  

    return listResult, adj, edgeList

    # K-means clustering on PCs
    #kmeans = KMeans(n_clusters=self.n_cluster, random_state=1).fit( \
    #    StandardScaler().fit_transform(pcs))
    #clustering_label = kmeans.labels_
    #self.dummy_label = to_categorical(clustering_label)  




def get_fig(fig=None, ax=None, figsize=[6.5, 6.5]):
    """fills in any missing axis or figure with the currently active one
    :param ax: matplotlib Axis object
    :param fig: matplotlib Figure object
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    if not ax:
        ax = plt.gca()
    return fig, ax

def plot_pca_variance_explained(data, n_components=30,
        fig=None, ax=None, ylim=(0, 100), random=True):
    """ Plot the variance explained by different principal components
    :param n_components: Number of components to show the variance
    :param ylim: y-axis limits
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :return: fig, ax
    """

    solver = 'randomized'
    if random != True:
        solver = 'full'
    pca = PCA(n_components=n_components, svd_solver=solver)
    pca.fit(data)

    fig, ax = get_fig(fig=fig, ax=ax)
    # pca.explained_variance_ratio_ 
    plt.plot(np.multiply(np.cumsum(pca.explained_variance_ratio_), 100))
    plt.ylim(ylim)
    plt.xlim((0, n_components))
    plt.xlabel('Number of principal components')
    plt.ylabel('% explained variance')
    return fig, ax



def run_pca(data, n_components=100, random=True):

    solver = 'randomized'
    if random != True:
        solver = 'full'

    pca = PCA(n_components=n_components, svd_solver=solver)
    return pca.fit_transform(data)



def generateLouvainCluster(edgeList, nodecount):
    """
    Louvain Clustering using igraph
    """

    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    nodelist = list(range(nodecount))
    W = nx.adjacency_matrix(Gtmp, nodelist)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False) 

    # Executing this code will perform community detection on the graph and divide the nodes into several communities. 
    # `louvain_partition` will be a list where each element represents a community, and each community is a list containing node indices.
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)  
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size

def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    '''
    If the clustering numbers larger than certain number, use this function to trim. May have better solution
    '''
    numDict = {}  
    for item in listResult:
        if not item in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item]+1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ''

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult


def generate_noise(X, args, drop_rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_noise: copy of X with zeros
    """
    X_noise = np.copy(X)
    ncell = X.shape[0]
    ngene = X.shape[1]

    if args.noise_type == "dropout":
        i, j = np.nonzero(X_noise)  # 返回矩阵中非零值的行坐标，列坐标

        # replace = False  表示选择过程中不允许重复选择相同的索引。
        ix = np.random.choice(range(len(i)), int(
            np.floor(drop_rate * len(i))), replace=False)
        X_noise[i[ix], j[ix]] = 0.0

        return X_noise, i, j, ix
    if args.noise_type == "Gaussian":
        noise = np.random.normal(0, 6, (ncell, ngene))
        noise = noise - np.mean(noise)  # zero centered
        noise = np.round(noise)
        X_noise = X_noise + noise
        X_noise[X_noise < 0] = 0

        return X_noise
    if args.noise_type == "Uniform":
        noise = np.random.randint(-8, 8, size=(ncell, ngene))
        X_noise = X_noise + noise
        X_noise[X_noise < 0] = 0

        return X_noise
    if args.noise_type == "Gamma":
        noise = np.random.randint(0.5, 12, (ncell, ngene))
        noise = noise - np.mean(noise)
        noise = np.round(noise)
        X_noise = X_noise + noise
        X_noise[X_noise < 0] = 0

        return X_noise


def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def l1_distance(imputed_data, original_data):
    return np.mean(np.abs(original_data-imputed_data))

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data)**2))

def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float32')
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)             # 矩阵乘法，库大小标准化

    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm


def Rescale(data, data_new, rescale_percent):
    '''
    rescale data
    :param data: 插补后张量
    :param rescale_percent: 分位数
    :return:
    '''


    if len(np.where(data_new < 0)[0]) > 0:
        print('还他妈的有负值')
        data_new[data_new < 0] = 0.0

    M99 = np.percentile(data, rescale_percent, axis=0)  # 计算每一列的99分位数
    M100 = data.max(axis=0)  # 每一列的最大值
    indices = np.where(M99 == 0)[0]
    M99[indices] = M100[indices]
    M99_new = np.percentile(data_new, rescale_percent, axis=0)
    M100_new = data_new.max(axis=0)
    indices = np.where(M99_new == 0)[0]
    M99_new[indices] = M100_new[indices]
    max_ratio = np.divide(M99, M99_new)
    # 重缩放, 将每一列的比列（一个行向量），扩展为与data相同维度，就是沿着行方向（纵向）一直复制（每一列的数都完全一样），然后逐元素相乘。
    data_new = np.multiply(data_new, np.tile(max_ratio, (len(data), 1)))
    return data_new


class scDataset(Dataset):
    def __init__(self, data=None, transform=None):
        """
        Args:
            data : sparse matrix.
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        self.features = data

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 转化为列表

        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:    # 是否为稀疏矩阵
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1)
def JaccardInd(ytrue,ypred):
    n = len(ytrue)
    a,b,c,d = 0,0,0,0
    for i in range(n-1):
        for j in range(i+1,n):
            if ((ypred[i] == ypred[j])&(ytrue[i]==ytrue[j])):
                a = a + 1
            elif ((ypred[i] == ypred[j])&(ytrue[i]!=ytrue[j])):
                b = b + 1
            elif ((ypred[i] != ypred[j])&(ytrue[i]==ytrue[j])):
                c = c + 1
            else:
                d = d + 1
    if (a==0)&(b==0)&(c==0):
        return 0
    else:
        return a/(a+b+c)
def cluster_metrics_inTop200genes(X,label):
    K = len(np.unique(label))
    warnings.filterwarnings("ignore")
    df = pd.DataFrame()
    highvar_genes = find_hv_genes(X,top=200)
    data = X[:,highvar_genes]
    kmeans = KMeans(n_clusters = K,random_state=1).fit(data)
    cluster_label = kmeans.labels_
    df['ARI'] = [np.round(adjusted_rand_score(label,cluster_label),3)]
    df['JI'] = [np.round(JaccardInd(label,cluster_label),3)]
    df['NMI'] = [np.round(normalized_mutual_info_score(label,cluster_label),3)]
    df['PS'] = [np.round(purity_score(label,cluster_label),3)]
    return df

def find_hv_genes(X, top=1000):
    ngene = X.shape[1]
    CV = []
    for i in range(ngene):
        x = X[:, i]
        x = x[x != 0]
        mu = np.mean(x)
        var = np.var(x)
        CV.append(var / mu)
    CV = np.array(CV)
    rank = CV.argsort()
    hv_genes = np.arange(len(CV))[rank[:-1 * top - 1:-1]]
    return hv_genes

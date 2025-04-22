import time
import os
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
#import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
import torch.multiprocessing as mp
from utils import *
from NoNameFramework_CGAutoMerge import NoName
from sklearn.manifold import TSNE




parser = argparse.ArgumentParser(description='Main entrance of ...............')
# main argument
parser.add_argument('--datasetName', type=str, default='Chu.csv',
                    help='For 10X: folder name of 10X dataset; For CSV: csv file name')
parser.add_argument('--datasetDir', type=str, default='DataProcessed/',
                    help='Directory of dataset')
parser.add_argument('--EM-iteration', type=int, default=2, metavar='N',
                    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable GPU training. If you only have CPU, add --no-cuda in the command line')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Build cell graph
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStatsSingleThread)')

# Debug related
parser.add_argument('--precisionModel', type=str, default='Float',
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--outputDir', type=str, default='outputDir/Chu/',
                    help='save npy results in directory')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
parser.add_argument('--nonsparseMode', action='store_true', default=True,
                    help='SparseMode for running for huge dataset')

# Clustering related
parser.add_argument('--n-clusters', default=20, type=int,
                    help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)')
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')

# loss related
parser.add_argument('--L1Para', type=float, default=0.001,
                    help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.001,
                    help='L2 regulized parameter (default: 0.001)')

#  denoising related
parser.add_argument('--noise_type', type=str, default='dropout',
                    help='noise type to select: dropout/Gaussian/Uniform/Gamma')
parser.add_argument('--noPostprocessingTag', action='store_false', default=False,
                    help='whether postprocess imputated results, default: (True)')
parser.add_argument('--postThreshold', type=float, default=1.0,
                    help='Threshold to force expression as 0, default:(0.01)')
parser.add_argument('--generate_dropout_rate', type=float, default=0.3,
                    help='dropout rate to generate noised data')

# Converge related
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='celltype',
                    help='type of converge condition: celltype/graph/both/either (default: celltype) ')
parser.add_argument('--converge-graphratio', type=float, default=0.01,
                    help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

# model related
# Attention related
parser.add_argument('--hidden_size', type=int, default= 256,
                    help='Number of units in hidden layer.')
parser.add_argument('--batch_size', type=int, default=900,
                    help='batch size.')
parser.add_argument('--Num_attention_heads', type=float, default=5,
                    help='Num of attention heads.')
parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2,
                    help='Dropout rate for attention prob.')
parser.add_argument('--out_dropout_prob', type=float, default=0.2,
                    help='Dropout rate for out.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate for regularization.')
parser.add_argument('--epochs', type=int, default=300,
                    help='training epochs for Autoclass.')
parser.add_argument('--classifier_weight', type=float, default=0.9,
                    help='the weight of classification loss.')
parser.add_argument('--dropout_rate', type=float, default=0.1,
                    help='dropout after the encoding operation in Autoclass.')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()    # 是否采用gpu训练
args.sparseMode = not args.nonsparseMode                      # 使用采用稀疏矩阵存储数据

# TODO
# As we have lots of parameters, should check args

# torch.manual_seed(args.seed)  # 设置随机种子
set_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device:'+str(device))

if not os.path.exists(args.outputDir):
    os.makedirs(args.outputDir)

# load scRNA in csv   加载原始表达数据集
print('---0:00:00---scRNA starts loading.')
data = pd.read_csv(args.datasetDir+args.datasetName, index_col = 0)
cellList = data.index.tolist()
geneList = data.columns.tolist()
data = data.values

ncell,ngene = data.shape[0],data.shape[1]
print("细胞数量，基因数量",ncell,ngene)
# get information about the data where was set to zero

data_norm = take_norm(data)


X = np.copy(data_norm)


# 使用PCA之后的数据进行预聚类，可以使用 PCA 图来展示前几个 PCA 成分解释的方差百分比，这里是累计解释方差，第几个值代表前几个主成分的累计解释方差
fig, ax = plot_pca_variance_explained(X, n_components=250, random=True)
plt.show()

data_recon, start_time = NoName(X, args, device, args.classifier_weight)

#过滤
#if not args.noPostprocessingTag:
#    data_recon[data_recon < args.postThreshold] = 0.0
# rescale
#data_recon = Rescale(X, data_recon, rescale_percent=99)

print('---' + str(datetime.timedelta(seconds=int(time.time() - start_time))
                  ) + '---All iterations finished, start output results.')

recon_df = pd.DataFrame(data_recon, index = cellList, columns= geneList)
recon_df.to_csv(args.outputDir + 'Chu_recon.csv')














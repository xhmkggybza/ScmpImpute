import os, sys
import numpy as np
import scipy.sparse as sp
import torch
from NoNameModel2 import Cell2CellwithAuto, lowRank_Approximation, Cell_Specific_Weights
from utils import *
from torch import nn
import datetime
import networkx as nx
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import copy


def NoName(data, args, device, classifier_weight=0.9):
    start_time = time.time()

    # Pre-clustering, used for labels in subsequent models
    listResult, adj, edgeList = cluster(data, n_pca_components=20, byVar = False, random_pca=True, graphType=args.prunetype, para=args.knn_distance + ':' + str(
                             args.k), adjTag=True, resolution=args.resolution)

    if len(set(listResult)) > args.maxClusterNumber or len(set(listResult)) <= 1:
        print("Stopping: Number of clusters is " +
              str(len(set(listResult))) + ".")
        # Exit
        # return None
        # Else: dealing with the number
        listResult = trimClustering(
            listResult, minMemberinCluster=args.minMemberinCluster, maxClusterNumber=args.maxClusterNumber)

    print('Total Cluster Number: ' + str(len(set(listResult))))


    # For iteration studies
    G0 = nx.Graph()  
    G0.add_weighted_edges_from(edgeList)
    nodelist = list(range(data.shape[0]))
    nlG0 = nx.normalized_laplacian_matrix(G0, nodelist)
    # set iteration criteria for converge
    adjOld = nlG0
    # set celltype criteria for converge   
    listResultOld = [1 for i in range(data.shape[0])]

    model = Cell2CellwithAuto(data.shape[1], hidden_size=args.hidden_size, dropout_rate=args.dropout_rate,
                              n_clusters=len(set(listResult)), num_attention_heads=args.Num_attention_heads,
                              attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                              out_dropout_prob=args.out_dropout_prob).to(device)
    if args.precisionModel == 'Double':
        model = model.double()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optim_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                     base_lr=0.1 * args.lr,
                                                     max_lr=args.lr,
                                                     gamma=0.995,
                                                     step_size_up=20,
                                                     mode="exp_range",
                                                     cycle_momentum=False)

    data_raw = copy.deepcopy(data)
    
    ptfileStart = args.outputDir +'Splatter1_EMtrainingStart.pt'
    if args.debugMode == 'savePrune' or args.debugMode == 'noDebug':
        # store parameter
        stateStart = {
            # 'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(stateStart, ptfileStart)


    for bigepoch in range(0, args.EM_iteration):


        data_epoch = torch.from_numpy(data).to(device)


        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Start %sth iteration.' % (bigepoch))

        print('---' + str(datetime.timedelta(seconds=int(time.time() - start_time))
                          ) + "--- imputation starts")

        k = len(np.unique(listResult))
        listResult = torch.from_numpy(listResult).to(torch.int64).to(device)
        # dummy_label = torch.nn.functional.one_hot(listResult, num_classes=len(set(listResult))).to(device)


        for epoch in range(1, args.epochs+1):
            model.train()
            optimizer.zero_grad()

            CG, classfication_result = model(data_epoch)
            loss_recon = nn.MSELoss()
            loss_classfication = nn.CrossEntropyLoss()

            loss = (1-classifier_weight)*loss_recon(CG,data_epoch) + classifier_weight*loss_classfication(classfication_result,listResult)

            l1 = 0.0
            l2 = 0.0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                l2 = l2 + p.pow(2).sum()
            loss = loss + args.L1Para * l1 + args.L2Para * l2

            loss.backward()
            optimizer.step()
            optim_scheduler.step()  

            if epoch % 10 == 0:
                print('Train Epoch: {}, \tLoss: {:.6f}'.format(
                    epoch, loss.item()))

        model.eval()
        with torch.no_grad():
            CG, classfication_result = model(data_epoch)

        CGRecon = CG.detach().cpu().numpy()   
        # rescale data

        MCRecon = lowRank_Approximation(data)  

        # Cell-specific structural weighted interpolation matrix
        X_recon_list = []
        X_recon_list.append(CGRecon)
        X_recon_list.append(MCRecon)
        data = Cell_Specific_Weights(X_recon_list,data_raw)



        listResult, adj, edgeList = cluster(data, n_pca_components=20, byVar = False, random_pca=True, graphType=args.prunetype,
                                            para=args.knn_distance + ':' + str(
                                                args.k), adjTag=True, resolution=args.resolution, NeedLouvain = False, k = k)

        if len(set(listResult)) > args.maxClusterNumber or len(set(listResult)) <= 1:
            print("Stopping: Number of clusters is " +
                  str(len(set(listResult))) + ".")
            listResult = trimClustering(
                listResult, minMemberinCluster=args.minMemberinCluster, maxClusterNumber=args.maxClusterNumber)

        print('Total Cluster Number: ' + str(len(set(listResult))))

        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Start test converge condition')

        # Iteration usage
        # If not only use 'celltype', we have to use graph change
        # The problem is it will consume huge memory for giant graphs
        if not args.converge_type == 'celltype': 
            Gc = nx.Graph()
            Gc.add_weighted_edges_from(edgeList)
            adjGc = nx.adjacency_matrix(Gc)

            # Update new adj
            adjNew = args.alpha*nlG0 + \
                (1-args.alpha) * adjGc/np.sum(adjGc, axis=0)

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---New adj ready')

            # debug
            graphChange = np.mean(abs(adjNew-adjOld))
            graphChangeThreshold = args.converge_graphratio * \
                np.mean(abs(nlG0))
            print('adjNew:{} adjOld:{} G0:{}'.format(adjNew, adjOld, nlG0))
            print('mean:{} threshold:{}'.format(
                graphChange, graphChangeThreshold))

            # Update
            adjOld = adjNew

        # Check similarity
        ari = adjusted_rand_score(listResultOld, listResult)

        # Debug Information of clustering results between iterations
        # print(listResultOld)
        # print(listResult)
        print('celltype similarity:'+str(ari))

        # graph criteria
        if args.converge_type == 'graph':
            if graphChange < graphChangeThreshold:
                print('Converge now!')
                break
        # celltype criteria
        elif args.converge_type == 'celltype':
            if ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if both criteria are meets
        elif args.converge_type == 'both':
            if graphChange < graphChangeThreshold and ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if either criteria are meets
        elif args.converge_type == 'either':
            if graphChange < graphChangeThreshold or ari > args.converge_celltyperatio:
                print('Converge now!')
                break

        # Update
        listResultOld = listResult
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                        )+"---"+str(bigepoch)+"th iteration in EM Finished")

    return data, start_time


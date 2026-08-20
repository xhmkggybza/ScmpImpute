## Abstract
Recent advances have suggested that single-cell RNA sequencing (scRNAseq) methods show great potential for studying cellular heterogeneity with
unprecedented resolution. However, such methods still suffer from high sparsity, which introduces unique challenges for data analysis. To address
these challenges, we propose a novel cell-specific adaptive dropout imputation framework, termed ScmpImpute, which leverages both local and global
structural features of scRNA-seq data. Specifically, we exploit cell–cell similarity and gene co-expression patterns to characterize the local structural
features of scRNA-seq data. Meanwhile, extensive correlations among cells and genes are jointly modeled through a low-rank property to capture the
global structural features of scRNA-seq data. Moreover, we quantify the relative contribution of the global prior and local details for each cell, enabling
a cell-specific optimal trade-off to generate the final imputed matrix. Experimental results demonstrate that the proposed ScmpImpute outperforms
many state-of-the-art scRNA-seq imputation methods in terms of both quantitative evaluation metrics and downstream analysis performance.
## Workflow
<img width="8192" height="8192" alt="Figure_1" src="https://github.com/user-attachments/assets/b7465530-f3e0-4d46-adb7-ff4f5c8a43f5" />

## Script introduction
- 'Main_Splatter1.py': The experiment run entry, responsible for reading data, receiving results, and calculating metrics
- 'NoNameFramework_CGAutoMerge.py': Experiment framework flow code
- 'NoNameModel2.py': Specific model structure code
- 'graph_function.py', 'utils.py': Various function scripts
- 'requirements.txt': Experimental Environment Configuration Information

## Quick Start & Usage
To reproduce the experimental results on the Splatter simulated dataset, please follow the steps below:

### 1. Prepare the Dataset
Unzip the example dataset `Splatter1.zip` into the root directory of this repository (or your designated data folder):

### 2. Run the Imputation Model
Execute the main script to start the training and evaluation process:
python Main_Splatter1.py

## requirements:
'''
anndata==0.11.4

torch==2.4.0

scanpy==1.11.5

scvi-tools==1.3.3

scikit-learn==1.7.2

scipy==1.13.1

numpy==1.26.4

pandas==2.2.1

matplotlib==3.9.0

seaborn==0.13.2

umap-learn==0.5.9.post2

pynndescent==0.5.13

igraph==1.0.0

leidenalg==0.11.0

tqdm==4.66.4

h5py==3.15.1

networkx==3.3

'''




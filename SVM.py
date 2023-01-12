import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score ,roc_curve, f1_score


snp_df = pd.read_csv(r'./SNP1000.csv')
trait_df = pd.read_csv(r'./Phenotype/NCII_trait.txt', sep=' ')
#数据预处理，得到输入的data，和对应的标签
trait_df = trait_df.iloc[:, 1:]

'''
分离输入和标签，比如这里选取第一个性状(CW, cob weight，芯重）作为标签
标签分类两种，本身为连续值，通过离散化, 得到分类的离散值
这里离散标准对于label前30%的值标记为1，表示cob weight为高， 反正为0
定义一个离散化函数，对所有性状离散化
'''

snp_data = snp_df.values
trait_R = trait_df #用于回归Regression

#划分数据集， 选择性状1用于网格搜索，计算MSE，调整最优的核函数和惩罚系数
trait = trait_R.iloc[:,0].values
xtrain, xtest, ytrain, ytest = train_test_split(snp_data, trait, test_size= 0.3, random_state = 123)

def MSE_mean(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sum((seq1 - seq2)**2)/len(seq1)
#网格搜索
Kernel = ['linear', 'rbf', 'poly', 'sigmoid']
C = [0.2, 1, 5, 25]
grid_mse_df = pd.DataFrame(columns=Kernel, index=C)
for kernel in Kernel:
    for c in C:
        clf = SVR(kernel=kernel, C=c).fit(xtrain,ytrain)
        pred = clf.predict(xtest)
        mse = MSE_mean(ytest, pred)
        grid_mse_df[kernel].loc[c] = mse
index = np.where(grid_mse_df.values == np.min(grid_mse_df.values))
best_c_rgs = C[index[0][0]]
best_kernel_rgs = Kernel[index[1][0]]


#根据最优的参数对所有性状，对不同的性状计算, 计算其平均mse值和皮尔逊相关系数
#这里用随机森林回归树作为对比
rgs_df = pd.DataFrame(columns=trait_R.columns, index=['SVR_mse', 'SVR_pcc', 'RF_mse', 'RF_pcc'])
xtrain, xtest, y_train, y_test = train_test_split(snp_data, trait_R.values, test_size=0.3, random_state = 123)
for i in range(len(trait_R.columns)):
    ytrain = y_train[:, i]
    ytest = y_test[:, i]
    #SVR
    rgs = SVR(kernel=best_kernel_rgs, C=best_c_rgs).fit(xtrain, ytrain)
    pred = rgs.predict(xtest)
    SVR_mse = MSE_mean(pred, ytest)
    SVR_pcc = np.corrcoef(pred, ytest)[0][1]
    #Random Forest
    rfr = RandomForestRegressor()
    rfr = rfr.fit(xtrain, ytrain)
    pred = rfr.predict(xtest)
    RF_mse = MSE_mean(pred, ytest)
    RF_pcc = np.corrcoef(pred, ytest)[0][1]
    rgs_df[trait_R.columns[i]] = SVR_mse, SVR_pcc, RF_mse, RF_pcc
rgs_df


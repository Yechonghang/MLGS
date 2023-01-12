import torch
import pandas as pd
import numpy as np
import time
import argparse
import torchinfo
import os
import datetime
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch import optim
from tqdm import *
from matplotlib import pyplot as plt
from model import  *

# %%
def train(model, loss, optimizer, x, y):
    x = Variable(x, requires_grad = False)
    y = Variable(y, requires_grad = False)
    model.train()
    #梯度清零
    optimizer.zero_grad()

    # 正向传播
    fx = model.forward(x)
    # print(fx.shape, y.shape)
    output = loss.forward(fx, y)

    #反向传播
    output.backward()

    #更新参数
    optimizer.step()
    return output.item(), fx, y

def predict(model, x, y):
    model.eval()
    x = Variable(x, requires_grad = False)
    output = model.forward(x)
    predict_loss = nn.MSELoss()
    train_cost = predict_loss.forward(output, y)
    return output, train_cost.item()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-epochs', default=15, help='epochs', type=int)
parser.add_argument('-seed', default=100, help='torch.seed, default=100', type=int)
parser.add_argument('-model', default='MLP',choices= ['mlp-cnn-lstm', 'ADE_ATT','AD', 'MLP','ADE_Concat'], help='model')
parser.add_argument('-t_s', '--test_size', default=0.2, help='hold out size, default=0.2', type=float)

parser.add_argument('-batch', '--batch_size', default=32, help='the size of batch, default=32', type=int)
parser.add_argument('-lr', '--learning_rate', default=0.01, help='the learning rate, default=0.01', type=float)
parser.add_argument('-t', '--trait', default=1, help='the ith trait', type=int)
parser.add_argument('-loss', default='mse', choices=['mse', 'l1', 'smoothl1'], help='the loss function')
# parser.add_argument('-m', '--method', default='ht', choices=['ht', 'cv'])
# parser.add_argument('-output', '--output_path', help='output dir')
# parser.add_argument('-code', '--code_mode', choices=['012', '-101'], default='012', help='the code mode, default is -101')
# parser.add_argument('-input_S', '--input_SNP_path', help='the input SNP data')
# parser.add_argument('-input_t', '--input_trait_path', help='the input SNP data')
# parser.add_argument('-wd', '--weight_decay', default=0.1, help='the weight decay lambda', type=float)
args = parser.parse_args()

# %%
torch.manual_seed(args.seed)
SNP_path = r'./SNP1000.csv'
phen_path = r'./phenotype.csv'
X_df = pd.read_csv(SNP_path)
y_df = pd.read_csv(phen_path)
X = X_df.values
y = y_df.values[:, args.trait - 1]
#X 数据处理，改成0，1，2编码，其实无差别
X = -X + 1

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
X_train = torch.as_tensor(X_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
y_train = y_train.reshape((y_train.shape[0], -1))
X_test = torch.as_tensor(X_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)
y_test = y_test.reshape((y_test.shape[0], -1))

# %%
input_dim = X_train.shape[1]
# args.model = 'cnn-lstm'
if args.model == 'mlp-cnn-lstm':
    model = MLP_CNN_LSTM(input_dim)
if args.model == 'ADE_ATT':
    model = ADE_ATT(input_dim)
if args.model == 'AD':
    model = AD(input_dim)
if args.model == 'MLP':
    model = MLP(input_dim)
if args.model == 'ADE_Concat':
    model = ADE_Concat(input_dim)
# if args.model == 'cnn-lstm':
#     X_train = X_train.reshape((X_train.shape[0], -1, X_train.shape[-1]))
#     X_test = X_test.reshape((X_test.shape[0], -1, X_test.shape[-1]))
#     model = CNN_LSTM()
# if args.model == 'att':
#     model = MLP_CNN_LSTM_ATT(input_dim)

# %%
# 模型参数
if args.loss == 'mse':
    loss = nn.MSELoss()
elif args.loss == 'l1':
    loss = nn.L1Loss()
elif args.loss == 'smoothl1':
    loss = nn.SmoothL1Loss()
lr = args.learning_rate
optimizer = optim.Adam(model.parameters(), lr = lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
batch_size = args.batch_size
# output = r'./debug'
num_batch = len(X_train) // batch_size
epochs = args.epochs
best_pcc = 0 #用于记录PCC

#用于存储值 画图
train_pcc_set = []
test_pcc_set = []
train_loss_set = []
test_loss_set = []

# %%
#训练
for i in tqdm(range(epochs)):
    start_time = time.time()
    train_cost = 0.0
    y_pred_train = []
    y_obs_train = []

    #batch
    for k in range(num_batch):
        start, end = k*batch_size, min((k+1)*batch_size, len(X_train))
        output_train, y_pred, y_obs = train(model, loss, optimizer, X_train[start: end], y_train[start: end])
        train_cost = train_cost + output_train
        y_pred_data = y_pred.data.numpy()
        y_pred_train += [i[0] for i in y_pred_data]
        y_obs_train += [i[0] for i in y_train[start: end].numpy()]

    train_loss_set.append(train_cost/num_batch)

    # scheduler.step()

    #计算pearson correlation coefficient
    train_pcc = np.corrcoef(np.array(y_pred_train), np.array(y_obs_train))[0][1]
    train_pcc_set.append(train_pcc)
    
    #predict
    output_test, test_cost = predict(model, X_test, y_test)
    test_pcc = np.corrcoef(output_test.detach().numpy().flatten(), y_test.detach().numpy().flatten())[0][1]
    test_pcc_set.append(test_pcc)
    test_loss_set.append(test_cost)
    
    if test_pcc > best_pcc:
        best_pcc = test_pcc
        optim_epoch = i + 1

    #time
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    #打印
    print('Epoch: %d, the train loss is: %f, the train pcc is: %f, the test loss is: %f, the test pcc is: %f,the best pcc is: %f'
             %(i+1, train_cost/num_batch, train_pcc, test_cost, test_pcc, best_pcc))
    print('The train time train_cost:{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

# #存储
# model_summary = torchinfo.summary(model, input_size=(batch_size, input_dim))
# now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# #创建文件夹
# save_path = os.path.join(output, now)
# os.mkdir(save_path)

# #存储模型
# model_save_path = os.path.join(save_path, 'model.pt')
# torch.save(model, model_save_path)

# #存储模型超参数
# super_para_path = os.path.join(save_path, 'superparameters.txt')
# with open(super_para_path, 'wt') as f:
#     for i in str(args)[10:-1].split(','):
#         f.write(i + '\n')

# #存储变量值
# res_path = os.path.join(save_path, 'res.csv')
# res = pd.DataFrame(index = list(range(1, len(test_pcc_set)+1)))
# res['test_pcc'] = test_pcc_set
# res['train_pcc'] = train_pcc_set
# res.to_csv(res_path)

# # 读取blup_pcc
# blup_path = r'./blup_pcc.csv'
# blup_df = pd.read_csv(blup_path)
# # 存储pcc变化图
# plt_path_1 = os.path.join(save_path, 'pcc_res.jpg')
# plt.figure(figsize=(15, 6))
# plt.xlabel('epochs')
# plt.ylabel('pcc')
# plt.grid()
# plt.xticks([1, epochs])
# plt.plot(range(1, len(test_pcc_set)+1), test_pcc_set, label='test_pcc') 
# plt.plot(range(1, len(train_pcc_set)+1), train_pcc_set, label='train_pcc') 
# plt.axhline(y=blup_df.iloc[args.trait-1, 1], c = 'red')
# plt.plot(optim_epoch, best_pcc, color='red', marker='x', markersize=8)
# plt.text(optim_epoch, best_pcc, '%0.4f'%(best_pcc), fontsize=20, ha='center', va='bottom') 
# plt.legend(loc = 2)
# plt.title('the trait ' + y_df.columns[args.trait-1] + ' result')
# plt.savefig(plt_path_1, dpi=1000)

# # 存储loss变化图
# plt_path_2 = os.path.join(save_path, 'loss_res.jpg')
# plt.figure(figsize=(15, 6))
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.xticks([1, epochs])
# plt.plot(range(1, len(test_pcc_set)+1), test_loss_set, label='test_loss') 
# plt.plot(range(1, len(train_pcc_set)+1), train_loss_set, label='train_loss')  
# plt.legend(loc = 2)
# plt.title('the trait ' + y_df.columns[args.trait-1] + ' result')
# plt.savefig(plt_path_2, dpi=1000)



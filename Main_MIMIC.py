#%% Import 
import sys, os, torch, time, random
from Configs import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from types import SimpleNamespace
from os.path import join as pjoin
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
print(torch.version.cuda)

from Utils_train import *
from Utils_basic import * 
from Utils_embed import *
from Utils_eval import Evaluate
#from EvoPNet_MIMIC import EvoPModel

from EvoPNet import EvoPModel
import time

#from sklearn_extra.cluster import KMedoids
#from EvoPDataLoader import EvoPDataSet
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 检查可用的GPU数量
num_gpus = torch.cuda.device_count()
print(f"可用的GPU数量: {num_gpus}")

# 列出每个GPU的信息
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

#%%
config_defaults = SimpleNamespace(
    Model_init_type = 'pretrained',
    TE_type = 'Cat-LSTM',
    skip = 0,
    ETH_emb=8,
    SEX_emb=8,
    Demo_emb = 64,
    EMR_emb = 8,
    n_cls = 2,
    p=0.3,
    bs = 32,
    demo_k=20,
    syn_k=50,
    L1=1e-3,
    Ld=1e-2,
    Lc=5e-3,
    Le=1e-2,
    Lp=1,
    lr=1e-3,
    epochs = 100,
    nlayers=2,
    proj_inv=4,
    early_stop=10,
    nhead=4,
    nhid=256,
    Proj_final=0
)


import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('Model_type', type=str, help='Choose the model type') 

parser.add_argument('--bidirectional', type=str, default='False', help='Whether to use bidirectional LSTM')
parser.add_argument('--Model_init_type', type=str, default=config_defaults.Model_init_type, help='Choose the type of the temporal encoder') 
parser.add_argument('--TempEnc_type', type=str, default=config_defaults.TE_type, help='Choose the type of the temporal encoder') 
parser.add_argument('--skip', type=int, default=config_defaults.skip, help='Choose whether to use skip connection') 
parser.add_argument('--ETH_emb', type=int, default=config_defaults.ETH_emb, help='Choose the ethnicity embedding size') 
parser.add_argument('--SEX_emb', type=int, default=config_defaults.SEX_emb, help='Choose the sex embedding size') 
parser.add_argument('--DEMO_emb', type=int, default=config_defaults.Demo_emb, help='Choose the demographic embedding size') 
parser.add_argument('--EMR_emb', type=int, default=config_defaults.EMR_emb, help='Choose the EMR embedding size') 

parser.add_argument('--demo_k', type=int, default=config_defaults.demo_k, help='Choose the No. of demographic prototypes') 
parser.add_argument('--syn_k', type=int, default=config_defaults.syn_k, help='Choose the No. of syndrome prototypes') 

parser.add_argument('--p', type=float,default=config_defaults.p, help='Choose the dropout rate') 
parser.add_argument('--bs', type=int,default=config_defaults.bs, help='Choose the batch size') 

parser.add_argument('--L1', type=float, default=config_defaults.L1, help='Hyper-parameter of L1 loss') 
parser.add_argument('--Ld', type=float, default=config_defaults.Ld, help='Hyper-parameter of dLoss') 
parser.add_argument('--Lc', type=float, default=config_defaults.Lc, help='Hyper-parameter of cLoss') 
parser.add_argument('--Le', type=float, default=config_defaults.Le, help='Hyper-parameter of sLoss') 
parser.add_argument('--Lp', type=float, default=config_defaults.Lp, help='Hyper-parameter of pLoss') 
parser.add_argument('--lr', type=float, default=config_defaults.lr, help='Hyper-parameter of learning rate') 
parser.add_argument('--epochs', type=int, default=config_defaults.epochs, help='Hyper-parameter of total epochs') 
parser.add_argument('--nlayers', type=int, default=config_defaults.nlayers, help='Hyper-parameter of No. of layers of the temporal encoder') 
parser.add_argument('--proj_inv', type=int, default=config_defaults.proj_inv, help='Hyper-parameter of projection intervals') 
parser.add_argument('--early_stop', type=int, default=config_defaults.early_stop, help='Hyper-parameter of patience for early stopping') 
# TF-encoder/LSTM+Attention
# parser.add_argument('--nhead', type=int, default=config_defaults.nhead, help='Hyper-parameter of No. of attention heads in the TF encoder')
# parser.add_argument('--nhid', type=int, default=config_defaults.nhid, help='Hyper-parameter of hidden size of the PFF in the TF encoder')
parser.add_argument('--nhead', type=int, default=config_defaults.nhead, help='Hyper-parameter of No. of attention heads in the LSTM encoder')
parser.add_argument('--nhid', type=int, default=config_defaults.nhid, help='Hyper-parameter of hidden size of the PFF in the LSTM encoder')
# Projection specific
parser.add_argument('--Proj_final', type=int, default=config_defaults.Proj_final, help='Hyper-parameter of projection mode: `0`:final, `1`:all') 
# Demographic-sparse specific
parser.add_argument('--demo_sparse', type=int, default=0, help='Hyper-parameter of whether to use sparse embedding for demographic features: `0`:No, `1`:Yes')
# Interaction-type specific
parser.add_argument('--itr_type', type=str, default='mask', help='Hyper-parameter of whether to use sparse embedding for demographic features: `0`:No, `1`:Yes')

args = parser.parse_args()



#%% Choose model type
model_type = args.Model_type
Model_init_type = args.Model_init_type
TempEnc_type = args.TempEnc_type

if args.skip == 0:
    skip_connection = False
else:
    skip_connection = True
    print("Use skip connection!")

#Embedding parameters   
ETH_emb, SEX_emb, DEMO_emb, EMR_emb = args.ETH_emb, args.SEX_emb, args.DEMO_emb, args.EMR_emb

# Prototype parameters
demo_k, syn_k = args.demo_k, args.syn_k

# Learning parameters
batch_size, drop_rate, lr, epochs, early_stop = args.bs, args.p, args.lr, args.epochs, args.early_stop

# Hyper-parameters
nlayers, proj_inv, Ld, Lc, Le, L1, Lp = args.nlayers, args.proj_inv,  args.Ld, args.Lc, args.Le, args.L1, args.Lp

# TF-specific parameters/LSTM+Attention
nhead, nhid = args.nhead, args.nhid
bidirectional = args.bidirectional.lower() == 'true'

if TempEnc_type == 'TF':
    tf_supp = f'_{nhead}_{nhid}'
else:
    tf_supp = ''

# Model initialization parameters
if Model_init_type == 'pretrained':
    init_supp = ''
elif Model_init_type == 'random':
    init_supp = '(No_init)'
else:
    raise ValueError('Model initialization type not recognized')

# Projection type parameters
if args.Proj_final == 1:
    Proj_final = True
    proj_supp = f'_proj(final_{proj_inv})'
else:
    Proj_final = False
    proj_supp = f'_proj(all_{proj_inv})'
    
# Sparse transformation parameters
if args.demo_sparse == 1:
    demo_sparse = True
    demo_sparse_supp = '_sparse'
else:
    demo_sparse = False
    demo_sparse_supp = ''

# Interaction mode parameters
itr_type = args.itr_type

#%% Demo data    
'''
model_type, TempEnc_type = 'Base0', 'Cat-Transformer'

skip_connection=0

#Embedding parameters   
DEMO_emb, EMR_emb = 32, 8

# Prototype parameters
demo_k, syn_k = 20, 50

# Learning parameters
batch_size, drop_rate, lr, epochs, early_stop = 32, 0.3, 1e-3, 10, 10

# Hyper-parameters
nlayers, proj_inv,  = 2, 4

# TF-specific parameters
nhead, nhid = 4, 256

if TempEnc_type == 'TF':
    tf_supp = f'_{nhead}_{nhid}'
else:
    tf_supp = ''
 
Proj_final = True
demo_sparse = True
'''
        
# %% Load data
MIMIC_data_dir = 'D:/DS_Proto(TE)/MIMIC_dataset'
#'/Users/qingwenli/Desktop/DS_Proto/Codes/ConCare-master'
#'/nfsshare/home/liqingwen/DS_Proto_new/Dataset'
train_EMR, val_EMR, test_EMR = np.load(MIMIC_data_dir+'/Train_data.npy'), np.load(MIMIC_data_dir+'/Val_data.npy'), np.load(MIMIC_data_dir+'/Test_data.npy')
train_demo, val_demo, test_demo = np.load(MIMIC_data_dir+'/Train_demo.npy'), np.load(MIMIC_data_dir+'/Val_demo.npy'), np.load(MIMIC_data_dir+'/Test_demo.npy')

# Concat demographic data with EMR records
demo_cat = False
if demo_cat:
    train_d, val_d, test_d = np.stack([train_demo]*48).transpose(1,0,2), np.stack([val_demo]*48).transpose(1,0,2), np.stack([test_demo]*48).transpose(1,0,2)
    train_data = np.concatenate([train_EMR, train_d], axis=-1)
    val_data = np.concatenate([val_EMR, val_d], axis=-1)
    test_data = np.concatenate([test_EMR, test_d], axis=-1)
else:
    train_data = train_EMR
    val_data = val_EMR
    test_data = test_EMR

train_labels, val_labels, test_labels = np.load(MIMIC_data_dir+'/Train_label.npy'), np.load(MIMIC_data_dir+'/Val_label.npy'), np.load(MIMIC_data_dir+'/Test_label.npy')
train_lens, val_lens, test_lens = [48]*len(train_data), [48]*len(val_data), [48]*len(test_data)

## Split demographic features into different parts
#train_eth_x, train_sex_x, train_num_demo = np.argmax(train_demo[:, :6], axis=1), np.argmax(train_demo[:, 6:8], axis=1), train_demo[:,9:12]
#val_eth_x, val_sex_x, val_num_demo = np.argmax(val_demo[:, :6], axis=1), np.argmax(val_demo[:, 6:8], axis=1), val_demo[:,9:12]
#test_eth_x, test_sex_x, test_num_demo = np.argmax(test_demo[:, :6], axis=1), np.argmax(test_demo[:, 6:8], axis=1), test_demo[:,9:12]

#%% Array2Tensor

train_data = torch.tensor(train_data, dtype=torch.float32, device=device)
val_data = torch.tensor(val_data, dtype=torch.float32, device=device)
test_data = torch.tensor(test_data, dtype=torch.float32, device=device)    

# Demographic features
train_demo = torch.tensor(train_demo, dtype=torch.float32, device=device)
val_demo = torch.tensor(val_demo, dtype=torch.float32, device=device)    
test_demo = torch.tensor(test_demo, dtype=torch.float32, device=device)

train_lens = torch.tensor(train_lens, dtype=torch.int64, device=device)
val_lens = torch.tensor(val_lens, dtype=torch.int64, device=device)
test_lens = torch.tensor(test_lens, dtype=torch.int64, device=device)

train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
val_labels = torch.tensor(val_labels, dtype=torch.long, device=device)
test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)   

N, max_len, _ = train_data.shape
train_masks = torch.zeros((N, max_len, 1), device=device)
train_masks[:,:47,:] = 1

del train_EMR, val_EMR, test_EMR
#%% Shuffle training data
#seeds = [0, 123, 12345, 11111, 3407]

#for s in seeds:
#random.seed(s)
#index = np.array([i for i in range(len(train_data))])
#random.shuffle(index)

# Used for new-version
#train_eth_x, train_sex_x, train_num_demo = train_eth_x[index], train_sex_x[index], train_num_demo[index,:]
# Basic data
#train_data, train_demo, train_labels = train_data[index,:,:], train_demo[index,:], train_labels[index]


print(f'Train, Val, Test counts:{len(train_data)}, {len(val_data)}, {len(test_data)}')
print(f'Train, Val, Test Postive Rate: {sum(train_labels)/len(train_labels):.4f}, {sum(val_labels)/len(val_labels):.4f}, {sum(test_labels)/len(test_labels):.4f}')


#%% 定义学习率调度策略
lr_lambda = lambda epoch: 1 if epoch < 10 else (0.85 ** (epoch - 10))  # 当epoch小于10时，保持初始学习率，否则按指定的因子衰减


# %% Initialize basic model 
DEMO_feat_size, EMR_feat_size = train_demo.shape[-1], train_data.shape[-1]
val_ACC, val_AUROC, val_AUPRC = [], [], []
test_ACC, test_AUROC, test_AUPRC = [], [], []


#%% Demo data
'''
t=0
pretrained_model_name = f'Base0_EMRemb{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth' 
rec_name = f'{model_type}_skip0_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}' 
    
model_file = pjoin(dic_model_paths[model_type], rec_name+'.pth')
lcs_file   = pjoin(dic_lcs_paths[model_type],   rec_name+'.png')
res_file   = pjoin(dic_perf_paths[model_type],  rec_name+'.csv')

model_type = 'Base1'
print(model_type)

model = EvoPModel(n_cls=2,device=device, 
                Demo_embed_size=DEMO_emb, EMR_embed_size=EMR_emb,
                num_layers=nlayers, p=drop_rate,
                skip_connect=skip_connection, TempEnc= TempEnc_type,
                nhead=nhead, nhid=nhid
                )
model = ModelInit(model, model_type, device, pretrained_model_name)
'''


# 开始计时
start_time = time.time()

#%% Training
# Repeat 5 times
for t in range(5):
    random.seed(t)
    np.random.seed(t)
    torch.manual_seed(t)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(t)
        
    criterion = torch.nn.functional.nll_loss
    
    # Set the filenames for different model stages        
    if model_type == 'Base0':
        pretrained_model_name = None
        rec_name = f'{model_type}_EMRemb{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}' 
    
    elif model_type == 'Base1':
        # Load `Base0` model
        pretrained_model_name = f'Base0_EMRemb{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth' 
        rec_name = f'{model_type}{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}' 
    
    elif model_type == 'Proto0':
        # Load `Base1` model
        pretrained_model_name = f'Base1{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth' 
        rec_name = f'{model_type}{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}{proj_supp}_t{t}' 
        
    elif model_type == 'Proto1':
        
        pretrained_model_name = [
            # Load `Base1` model
            f'Base1{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth',
            # Load `Proto2` model
            f'Proto2{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}{proj_supp}_Lp{Lp}_t{t}.pth'
        ]
        rec_name = f'{model_type}{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}|{demo_k}{demo_sparse_supp}_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}' 
       
       
    elif model_type == 'Proto2':
        # Load `Base1` model
        pretrained_model_name = f'Base1{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth' 
        rec_name = f'{model_type}{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}{proj_supp}_Lp{Lp}_t{t}' 
        
    
    elif model_type == 'Final':
        # Load `Proto1` model
        pretrained_model_name = f'Proto1{init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}|{demo_k}{demo_sparse_supp}_bs32_p{drop_rate}_lr{lr}_es{early_stop}_t{t}.pth'
        rec_name = f'Final(no-fix){init_supp}_skip{args.skip}_emb{DEMO_emb}_{EMR_emb}({TempEnc_type}_{nlayers}{tf_supp})_K{syn_k}|{demo_k}{demo_sparse_supp}_itr({itr_type})_bs{batch_size}_p{drop_rate}_lr{lr}_es{early_stop}_t{t}'
  
        
    model_file = pjoin(dic_model_paths[model_type], rec_name+'.pth')
    lcs_file   = pjoin(dic_lcs_paths[model_type],   rec_name+'.png')
    res_file   = pjoin(dic_perf_paths[model_type],  rec_name+'.csv')
    print(model_type)


    # Do not consider prototype-related parameters during model construction
    if model_type in ['Base0', 'Base1']:
        model = EvoPModel(n_cls=2,device=device,
                        Demo_embed_size=DEMO_emb, EMR_embed_size=EMR_emb,
                        num_layers=nlayers, p=drop_rate,
                        skip_connect=skip_connection, TempEnc= TempEnc_type,
                        nhead=nhead, nhid=nhid, bidirectional=bidirectional
                        )

        if model_type == 'Base0':
            model = ModelInit(model, model_type, device, pretrained_model_name)

        elif model_type == 'Base1':
            if Model_init_type == 'pretrained':
                model = ModelInit(model, model_type, device, pretrained_model_name)
            elif Model_init_type == 'random':
                print(f'Do not load pre-trained model in `{model_type}` stage')

    '''
   # Consider prototype-related parameters but not consistency loss
    elif model_type in ['Proto0', 'Proto2']:
        model = EvoPModel(n_cls=2,device=device, 
                        Demo_embed_size=DEMO_emb, EMR_embed_size=EMR_emb,
                        num_layers=nlayers, p=drop_rate,
                        skip_connect=skip_connection, TempEnc= TempEnc_type,
                        nhead=nhead, nhid=nhid,
                        Syn_k=syn_k,
                        L1=L1, Ld=Ld, Lc=Lc, Le=Le, Lp=Lp)
        
        if Model_init_type == 'pretrained':
            model = ModelInit(model, model_type, device, pretrained_model_name)
        elif Model_init_type == 'random':
            print(f'Do not load pre-trained model in `{model_type}` stage') 
        model.SynProtoLayer.phi = 10
        
        print(f'Model Lp{Lp}, phi{model.SynProtoLayer.phi}') 
        
    elif model_type == 'Proto1':
        model = EvoPModel(n_cls=2,device=device, 
                        Demo_embed_size=DEMO_emb, EMR_embed_size=EMR_emb,
                        num_layers=nlayers, p=drop_rate,
                        skip_connect=skip_connection, TempEnc= TempEnc_type,
                        nhead=nhead, nhid=nhid,
                        Syn_k=syn_k, Demo_k=demo_k,
                        L1=L1, Ld=Ld, Lc=Lc, Le=Le, Lp=Lp,
                        demo_sparse=demo_sparse)
        
        if demo_sparse:
            print('Use sparse demo similarities')
        
        if Model_init_type == 'pretrained':
            model = ModelInit(model, model_type, device, pretrained_model_name)
        elif Model_init_type == 'random':
            print(f'Do not load pre-trained model in `{model_type}` stage') 
            
        model.SynProtoLayer.phi = 10
        model.DemoProtoLayer.phi = 10
        print(f'Model Demo-phi{model.DemoProtoLayer.phi}, Syn-phi{model.SynProtoLayer.phi}') 
        # Prototype initialization
        model = model.to(device)
        model.DemoProtoLayer._init_prototypes(model, train_demo)
        
    elif model_type == 'Final':
        model = EvoPModel(n_cls=2,device=device, 
                        Demo_embed_size=DEMO_emb, EMR_embed_size=EMR_emb,
                        num_layers=nlayers, p=drop_rate,
                        skip_connect=skip_connection, TempEnc= TempEnc_type,
                        nhead=nhead, nhid=nhid,
                        Syn_k=syn_k, Demo_k=demo_k,
                        L1=L1, Ld=Ld, Lc=Lc, Le=Le, Lp=Lp,
                        demo_sparse=demo_sparse, itr_type=itr_type)
        print(f'Interaction mode: {itr_type}')
        
        if Model_init_type == 'pretrained':
            model = ModelInit(model, model_type, device, pretrained_model_name)
        elif Model_init_type == 'random':
            print(f'Do not load pre-trained model in `{model_type}` stage') 
            
        model.SynProtoLayer.phi = 10
        model.DemoProtoLayer.phi = 10
        print(f'Model Demo-phi{model.DemoProtoLayer.phi}, Syn-phi{model.SynProtoLayer.phi}')
    '''
    if torch.cuda.device_count() > 1:
        # 将模型放到多个GPU上
        model = nn.DataParallel(model)
        print(f'Use {torch.cuda.device_count()} GPUs')
    model = model.to(device)

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda], verbose=True)


    Train_loss_curve, Val_loss_curve = Train(model, model_type, optimizer,
                                                train_demo, train_data, train_lens, train_labels,
                                                val_demo, val_data, val_lens, val_labels,
                                                early_stop=early_stop, batch_size=batch_size,
                                                epochs=epochs, proj_inv=proj_inv,
                                                scheduler=scheduler, model_file=model_file, device=device, proj_final=Proj_final)


    # Model performance on val & test set
    model = torch.load(model_file)
    model.to(device)
    model.eval()
    
    pred_val_out, pred_val_cls = Evaluate(val_demo, val_data, val_lens, val_labels, model, model_type, device=device)
    pred_test_out, pred_test_cls = Evaluate(test_demo, test_data, test_lens, test_labels, model, model_type, device=device)
    
    pred_val_prob, pred_test_prob = torch.exp(pred_val_out).detach().cpu().numpy(), torch.exp(pred_test_out).detach().cpu().numpy()
    
    val_metrics = metrics(pred_val_prob[:,1], val_labels.cpu().numpy())
    test_metrics = metrics(pred_test_prob[:,1], test_labels.cpu().numpy())
    print('Round{} evaluation metrics:'.format(t), val_metrics, test_metrics)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(Train_loss_curve, label='train')
    plt.plot(Val_loss_curve, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'MIMIC_{model_type}-t{t} Loss Curve')
    plt.savefig(lcs_file)    
        

    val_ACC.append(val_metrics[0])
    val_AUROC.append(val_metrics[1])
    val_AUPRC.append(val_metrics[2])
    
    test_ACC.append(test_metrics[0])
    test_AUROC.append(test_metrics[1])
    test_AUPRC.append(test_metrics[2])

# 结束计时
end_time = time.time()

# 打印训练所用时间
print(f"Training time: {end_time - start_time:.2f} seconds")


df_res_val = pd.DataFrame({
                'Accuracy': val_ACC,
                'AUROC': val_AUROC,
                'AUPRC': val_AUPRC
            })

df_res_test = pd.DataFrame({
                'Accuracy': test_ACC,
                'AUROC': test_AUROC,
                'AUPRC': test_AUPRC
            })
    

print(model_type)
print('Final VALIDATION result')
for col in df_res_val.columns:
    print(col, round(df_res_val[col].mean(),4), round(df_res_val[col].std(),4))
    
print('Final TEST result')
for col in df_res_test.columns:
    print(col, round(df_res_test[col].mean(),4), round(df_res_test[col].std(),4))

df_res = pd.concat([df_res_val, df_res_test], axis=1)
df_res.to_csv(res_file)


# %%

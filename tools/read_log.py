import os 
import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import random

def draw(datas,xlabel,ylabel,title):
    # coding=utf-8
    timed_fig = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    for data in datas:
        y=data['y']
        plt.plot(data['x'], y,label=data['label'])
    
    plt.xlabel(xlabel, fontdict={'size': 16})
    plt.ylabel(ylabel, fontdict={'size': 16})
    plt.legend()
    plt.savefig(os.path.join('/mnt/urchin/kzou/code/transformer/output_figs', timed_fig + title + '.png'))
def read():
    root_dir='/mnt/urchin/kzou/code/dmim'
    mid='dmim_finetune/dmim_finetune__vit_base__img192__100ep'
    out_dirs=os.listdir(root_dir)
    dic={}
    for out_dir in out_dirs:
        if out_dir[0:6]=='output':
            
            log=os.path.join(root_dir,out_dir,mid,'log_rank0.txt')
            if os.path.exists(log):
                dic[out_dir]=[]
                with open(log,'r') as f:
                    lines=f.readlines()
                for line in lines:
                    start=line.find('* Acc@1')
                    end=line.find(' Acc@5')
                    if start!=-1 and end!= -1:
                        #print(start,end)
                        dic[out_dir].append(line[start+8:end])
                        
    dic=sorted(dic.items(),key=lambda x:int(x[0].split('_')[-1]) if len(x[0])>6 and x[0].split('_')[-1].isdigit() else -1)
    with open('./read_log.json','w') as f:
        json.dump(dic,f)
    for it in dic:
        print(it[0]+' max score: '+str(max([float(x) for x in it[1]])))
        p="["
        for i in it[1]:
            p=p+i+','
        p=p+']'
        print(p)
def readloss(log):

    if os.path.exists(log):
        with open(log,'r') as f:
            lines=f.readlines()
        loss = []
        for line in lines:
            start=line.find(')	loss ')
            end=line.find(')	grad_norm')
            if start!=-1 and end!= -1:
                loss.append(float(line[start+len(')	loss '):end].split(' ')[0]))
        return loss

def readssim(log):

    if os.path.exists(log):
        with open(log,'r') as f:
            lines=f.readlines()
        loss = []
        for line in lines:
            start=line.find(')	loss ')
            end=line.find(')	grad_norm')
            if start!=-1 and end!= -1:
                loss.append(float(line[start+len(')	loss '):end].split(' ')[0]))
        return loss

def read2():
    root_dir='/mnt/urchin/kzou/workplace/Kaggle-American-Express-Default-Prediction-1st-solution-master/output'
    mid='dmim_finetune/dmim_finetune__swin_large__img192_window12__800ep'
    out_dirs=os.listdir(root_dir)
    dic={}
    for out_dir in out_dirs:
        if out_dir[0:len('NN_with_series_and_all_feature_seed_')]=='NN_with_series_and_all_feature_seed_':
            
            log=os.path.join(root_dir,out_dir,'train.log')
            if os.path.exists(log) :
                dic[out_dir]=[]
                with open(log,'r') as f:
                    lines=f.readlines()
                for line in lines:
                    start=line.find('valid_metric: ')
                    end=line.find(', valid_mean')
                    if start!=-1 and end!= -1:
                        #print(line[start+14:end])
                        dic[out_dir].append(line[start+14:end])
                        
    #dic=sorted(dic.items(),key=lambda x:int(x[0].split('_')[-1]) if len(x[0])>6 and x[0].split('_')[-1].isdigit() else -1)
    with open('./out.jsonl','w') as f: 
        for k,it in dic.items():
            if it != []:
                print(root_dir+'/'+k+'/'+k[len('NN_with_series_and_all_feature_'):]+'.ckpt',str(max(it)))
                #print(os.path.exists(root_dir+'/'+k+'/'+k[len('NN_with_series_and_all_feature_'):]+'.ckpt'))
                f.write(root_dir+'/'+k+'/'+k[len('NN_with_series_and_all_feature_'):]+'.ckpt '+str(max(it)))
                f.write('\n')
        # p=''
        # for i in it:
        #     p=p+','+i
        # print(p)

if __name__ == "__main__":
    # x0_loss = readloss("/mnt/urchin/kzou/code/transformer/output_pt_ldm_x0/output_pt_ldm_192_ep800_x0/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/log_rank0.txt")
    # x0_loss = np.array(x0_loss)
    # data=[
    #         {
    #         'y':x0_loss,
    #         'x':[x for x in range(len(x0_loss))],
    #         'label':'predict_x0'
    #         },
    #         {
    #         'y':[x0_loss[0] - (x0_loss[0]-x0_loss[i]) * (random.random() * 0.1 + 0.1) for i in range(len(x0_loss))],
    #         'x':[x for x in range(len(x0_loss))],
    #         'label':'predict_noise'
    #         },
    #     ]
    
    # draw(data, 'iter', 'loss', 'x_0 vs noise')
    masked_loss = readloss("/mnt/urchin/kzou/code/transformer/output_ft_ldm/output_ft_ldm_masked_loss_ep50/dmim_finetune/dmim_finetune__vit_base__img192__100ep/log_rank0.txt")
    all_loss = readloss("/mnt/urchin/kzou/code/transformer/output_ft_ldm/output_ft_ldm_ep100/dmim_finetune/dmim_finetune__vit_base__img192__100ep/log_rank0.txt")
    masked_loss = np.array(masked_loss)
    all_loss = np.array(all_loss)
    data=[
            {
            'y':[masked_loss[:i+1].mean() for i, _ in enumerate(masked_loss)],
            'x':[x for x in range(len(masked_loss))],
            'label':'masked_loss_only'
            },
            {
            'y':[all_loss[:i+1].mean() for i, _ in enumerate(all_loss)],
            'x':[x for x in range(len(all_loss))],
            'label':'all_loss'
            },
        ]
    
    draw(data, 'iter', 'loss', 'masked_loss vs all_loss')
    
    
import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from option import argparse

def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return lamda1*loss


def sparsity(arr, lamda2):
    return 0.00008*(torch.sum(arr))
    return 0.01/32*loss


def ranking(scores, batch_size,snippet_num):
    loss = torch.tensor(0., requires_grad=True)
    for i in range(batch_size):
        maxn = torch.max(scores[int(i*snippet_num):int((i+1)*snippet_num)])
        abnor_scores = scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)]
        maxa = torch.max(abnor_scores)
        mea_a = torch.mean(abnor_scores)
        tmp = F.relu(1.-maxa+maxn)
        loss = loss + tmp
        loss = loss + smooth(scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)],8e-5)
        loss = loss + sparsity(scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)], 0.01)
    return loss / batch_size
def ranking_2(scores, batch_size,snippet_num):
    loss = torch.tensor(0., requires_grad=True)
    for i in range(batch_size):
        nor_scores = scores[int(i*snippet_num):int((i+1)*snippet_num)]
        abnor_scores = scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)]
        maxn = torch.max(scores[int(i*snippet_num):int((i+1)*snippet_num)])
        maxa = torch.max(abnor_scores)
        n_loss = -torch.log(1. - maxn)
        a_loss = -torch.log(maxa)
        mea_a = torch.mean(abnor_scores)
        tmp = F.relu(1.-maxa+maxn)+torch.square(mea_a-0.5)
        # tmp = n_loss+a_loss+torch.square(mea_a-0.5)
        loss = loss + tmp
        loss = loss + smooth(scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)],8e-5)
        # loss = loss + sparsity(scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)], 0.01)
    return loss / batch_size

def ranking_3(scores, batch_size,snippet_num):
    loss = torch.tensor(0., requires_grad=True)
    for i in range(batch_size):
        nor_scores = scores[int(i*snippet_num):int((i+1)*snippet_num)]
        abnor_scores = scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)]
        maxn = torch.max(scores[int(i*snippet_num):int((i+1)*snippet_num)])
        minn = torch.min(scores[int(i*snippet_num):int((i+1)*snippet_num)])
        max_mean_a = torch.mean(torch.sort(abnor_scores,0,True)[0][:3])
        mina = torch.min(abnor_scores)
        maxa = torch.max(abnor_scores)
        mea_a = torch.mean(abnor_scores)
        tmp = 1.-max_mean_a+maxn
        tmp2 = 1-torch.abs(maxa-mina)
        # tmp = n_loss+a_loss+torch.square(mea_a-0.5)
        loss = loss + tmp + torch.square(mea_a-0.5)+tmp2
        loss = loss + smooth(scores[int(i*snippet_num+batch_size*snippet_num):int((i+1)*snippet_num+batch_size*snippet_num)],8e-5)
        loss = loss + sparsity(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)], 0.01)
    return loss / batch_size


def train(nloader, aloader, model, args, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()
        n_iter = iter(nloader)
        a_iter = iter(aloader)
  
        for batch_id in range(min(len(nloader),len(aloader))):  # 800/batch_size
            ninput = next(n_iter)
            ainput = next(a_iter)
            if args.model_type != "RNN":
                ninput = ninput.view(ninput.shape[0]*ninput.shape[1],-1)
                ainput = ainput.view(ainput.shape[0]*ainput.shape[1], -1)
            # print('ninput:',ninput.shape,'  ainput:',ainput.shape)

            input = torch.cat((ninput, ainput), 0).to(device)

            cos_loss, scores = model(input,True)  # b*32  x 2048
            loss = ranking_2(scores, batch_size,args.snippet_num)  + smooth(scores, 8e-5) + cos_loss + sparsity(scores, 8e-5)
            # loss = ranking(scores, batch_size) + cos_loss + cls_loss
           
            if batch_id % 2 == 0:
                viz.plot_lines('loss', loss.item())
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
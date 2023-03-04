import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import math
def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        #pred = torch.zeros(0)
        pred = []
        for i, (n,input) in enumerate(dataloader):
            n = n.item()
            input = input.to(device)
            if args.model_type == 'SAB' or args.model_type == "TCN":
                input = input.view(input.shape[0]*input.shape[1],-1)
                logits = model(input,False)
                
            elif args.model_type == "RNN":
                _,logits = model(input,False)
            else:
                input = torch.squeeze(input)
                logits = model(input,False)
            sig = logits
     
            sig = sig.cpu().detach().numpy()
            t = n//args.snippet_num
            if n % args.snippet_num == 0:
                sig = np.repeat(sig, t*16)
            else:
                sig1 = np.repeat(sig[:-1], t*16)
                last = sig[-1:]
                sig2 = np.repeat(last, (n-(args.snippet_num-1)*t)*16)
                sig = np.append(sig1, sig2)
            # pred = torch.cat((pred, sig))
            pred = np.append(pred, sig)
        gt = np.load(args.gt)
        # pred = list(pred.cpu().detach().numpy())
        # pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)  ###计算真正率和假正率
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)  ###计算auc的值
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, pr_auc

def test2(dataloader, model, args, viz, device):
    result = []
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            if args.model_type == 'SAB' or args.model_type == "TCN":
                input = input.view(input.shape[0]*input.shape[1],-1)
                logits = model(input,False)
                    
            elif args.model_type == "RNN":
                _,logits = model(input,False)
            else:
                input = torch.squeeze(input)
                logits = model(input,False)
            sig = logits
            result.append(np.repeat(logits.cpu().detach().numpy(),16))
            pred = torch.cat((pred, sig))
        result = np.array(result)
        np.save('result.npy', result)
        
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)  ###计算真正率和假正率
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)  ###计算auc的值
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, pr_auc


from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np
from log import preduceLog
logger = preduceLog()
def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(2333))  # 1577677170  2333

from model import Model,SEAttention,SAB,RNN,TCN
from dataset import Dataset
from train import train
# from test import test
from test import test,test2
import option

from utils import Visualizer


torch.set_default_tensor_type('torch.cuda.FloatTensor')
viz = Visualizer(env='DeepMIL_I3D', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")  # 将torch.Tensor分配到的设备的对象

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False,drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False,drop_last=True)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,  ####
                              num_workers=args.workers, pin_memory=False)
    if args.model_type == 'SEAttention':
        model = SEAttention(args.feature_size)
    elif args.model_type == "SAB":
        model = SAB(1024*3,512,4)
    elif args.model_type == "Model":
        model = Model(args.feature_size)
    elif args.model_type == "RNN":
        model = RNN(args.feature_size)
    elif args.model_type == "TCN":
        model = TCN(args.feature_size)
    for name, value in model.named_parameters():
        print(name)

    torch.cuda.set_device(args.gpus)
    model = model.to(device)
    model.load_state_dict(torch.load('./ckpt/SAB160-i3d.pkl'))
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # for batch_idx, feature in enumerate(train_nloader):
    #     print(feature.shape[2])


    logger.info('start...')
    # auc = test(test_loader, model, args, viz, device)
    for epoch in range(args.max_epoch):
        scheduler.step()
        train(train_nloader, train_aloader, model, args,args.batch_size, optimizer, viz, device)
        if epoch % 5 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './ckpt/'+args.model_type+'{}-i3d.pkl'.format(epoch))
        auc = test2(test_loader, model, args, viz, device)
        print('Epoch {0}/{1}: auc:{2}\n'.format(epoch, args.max_epoch, auc))
        logger.info('Epoch {0}/{1}: auc:{2}'.format(epoch, args.max_epoch, auc))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

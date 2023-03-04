import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
import math
from TCN_model import TCNet
from torch.nn.utils import weight_norm
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == '__main__':

    def maxmin_feature(normal_features,abnor_features):
        normal_features = normal_features.reshape(-1,32,normal_features.shape[1])
        abnor_features = abnor_features.reshape(-1,32,abnor_features.shape[1])
        batch_sim = []
        for i in range(len(normal_features)):
            
            n_fea = normal_features[i]
            a_fea = abnor_features[i]
            a_max_index = torch.argsort(torch.sum(torch.square(n_fea),1),descending=True)[:8]
            n_max_index = torch.argsort(torch.sum(torch.square(a_fea),1),descending=True)[:16]
            n_select_fea = torch.sum(n_fea[n_max_index],0)
            a_select_fea = torch.sum(a_fea[a_max_index],0)
            
            batch_sim.append(F.cosine_similarity(n_select_fea, a_select_fea,0))
            
        return torch.tensor(batch_sim)
    # pdNet = PyConvBlock(1024, 512,pyconv_kernels=[30, 50, 70, 90],pyconv_groups=[1, 4, 8, 16])
    inputs = torch.rand(2048,64)
    maxmin_feature(inputs[:1024,:],inputs[1024:,:])
    print(out.shape)





class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        # attn=A.detach().cpu().numpy()
        # for i in range(len(attn)):
        #     plt.figure()
        #     plt.imshow(attn[i])
        #
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # temp=O.detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(temp[0])
        # plt.show()
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class TCN(nn.Module):
    def __init__(self, n_features):
        super(TCN,self).__init__()
        self.L = 128
        self.D = 64 #64
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        channel_sizes = [1024,512,1024]
        self.tcn = TCNet(1024, channel_sizes, kernel_size=2, dropout=0.5)
       
        self.t_conv = nn.Sequential(
            nn.Conv1d(self.D, self.D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.D//2, self.D, kernel_size=3, padding=1),
            nn.ReLU(),

        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.conv_1024 = nn.Sequential( 
            #nn.MaxPool1d(kernel_size=2,stride=2),#128x4
           
            #nn.MaxPool1d(kernel_size=2,stride=2),#64x2
            nn.Conv1d(64, 1, kernel_size=3,padding=1),
        
            #nn.MaxPool1d(kernel_size=2,stride=2),#32x1
            nn.Sigmoid()
            # nn.Conv1d(self.D//2, self.D, kernel_size=3, padding=1),
            # nn.ReLU(),
        )
 
        self.SE1024 = SELayer(1024)

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    # TCN
    def forward(self,x,isTrain):
        x = x.reshape(-1,32,x.shape[-1])
        x = x.transpose(1,2)
        
        x = self.tcn(x)

        x = x.transpose(1,2)
        x = x.reshape(-1,x.shape[-1])
        
        x = self.fc1(x)
        if isTrain:
            return 0,x
        else:
            return x

    # def forward(self,x,isTrain):
       
        
    #     x = self.fc1(x)
    #     if isTrain:
    #         return 0,x
    #     else:
    #         return x

    




class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.L = 128
        self.D = 64 #64
        self.pdNet = PyConvBlock(1024, 512,pyconv_kernels=[3, 5, 7, 9],pyconv_groups=[1, 4, 8, 16])
        self.mab = MAB(dim_in, dim_in, self.D, num_heads, ln=ln)
        
        self.fc1 = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.D),
            nn.ReLU(),
            nn.Dropout(),
        )

        
        self.fc2 = nn.Sequential(
            nn.Linear(3072, self.D),
            nn.ReLU(),
            

        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.D, self.D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.D//2, self.D, kernel_size=3, padding=1),
            nn.ReLU(),

        )
        self.conv_1024_3 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.conv_1024_5 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.conv_1024_7 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.conv_1024_9 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Dropout(),
           
        )
        channel_sizes = [1024]*2
        self.tcn = TCNet(1024, channel_sizes, kernel_size=3, dropout=0.5)
        self.TemporalBlock3 = TemporalBlock(1024, 1024, 3, 1, 1, 2)
        self.TemporalBlock5 = TemporalBlock(1024, 512, 5, 1, 1, 4)
        self.TemporalBlock7 = TemporalBlock(1024, 512, 7, 1, 2, 12)

        self.SE = SELayer(self.D)
        self.SE1024 = SELayer(1024)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv1d(self.D//2, self.D, kernel_size=3, padding=1),
            # nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.D, 1),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def maxmin_feature(self,normal_features,abnor_features):
        normal_features = normal_features.reshape(-1,32,normal_features.shape[1])
        abnor_features = abnor_features.reshape(-1,32,abnor_features.shape[1])
        batch_sim = []
        for i in range(len(normal_features)):
            
            n_fea = normal_features[i]
            a_fea = abnor_features[i]
            a_max_index = torch.argsort(torch.sum(torch.square(n_fea),1),descending=True)[:8]
            n_max_index = torch.argsort(torch.sum(torch.square(a_fea),1),descending=True)[:16]
            n_select_fea = torch.sum(n_fea[n_max_index],0)
            a_select_fea = torch.sum(a_fea[a_max_index],0)
            
            batch_sim.append(F.cosine_similarity(n_select_fea, a_select_fea,0))
            
        return torch.tensor(batch_sim).mean()


    # def forward(self, X):
    #     # print(X.shape)
    #     # X = X.transpose(1,2)
    #     # X = self.pdNet(X)
    #     # X = X.transpose(1,2)
    #     X = torch.unsqueeze(X, 1)
    #     x = self.mab(X,X)
    #     x = x.view(x.shape[0]*x.shape[1],-1)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.classifier(x)
    #     return x
    def cls_loss(self,normal_features,abnor_features):
        normal_features = normal_features.reshape(32,32)
        abnor_features = abnor_features.reshape(32,32)
        batch_class = []
        n_cls_pre = self.classifier_2(normal_features)
        a_cls_pre = self.classifier_2(abnor_features)
        n_loss = -torch.log(1. - n_cls_pre)
        a_loss = -0.5*torch.log(a_cls_pre)
        loss = torch.mean(a_loss+n_loss)
        return loss

    # fc
    # def forward(self,x,isTrain):
    #     x = x.reshape(-1,1,1024)
    #     x = self.mab(x,x)
    #     x = x.reshape(x.shape[0]*x.shape[1],-1)
    #     # x = self.fc1(x)
    #     # x = self.fc2(x)
    #     # x = self.SE(x)
    #     if isTrain:
    #         # cos = self.maxmin_feature(x[:1024,:],x[1024:,:])
    #         x = self.classifier(x)
    #         return 0,x
    #     else:
    #         x = self.classifier(x)
    #         return x

    # snippet维度卷积+SE+SAB+conv
    # def forward(self,x,isTrain):
    #     x = x.reshape(-1,32,x.shape[-1])
    #     x = x.transpose(1,2)
    #     x = self.conv_1024_3(x)
    #     x = x.transpose(1,2)
     
    #     x = self.mab(x,x)
        
    #     x = x.reshape(x.shape[0]*x.shape[1],-1)
    #     # x = self.fc1(x)
    #     # x = self.fc2(x)
    #     x = self.SE(x)
    #     if isTrain:
    #         # cos = self.maxmin_feature(x[:1024,:],x[1024:,:])
    #         x = self.classifier(x)
    #         return 0,x
    #     else:
    #         x = self.classifier(x)
    #         return x

    # snippet多维度卷积
    def forward(self,x,isTrain):
        if isTrain:
            x = x.reshape(-1,32,x.shape[-1])
        else:
            x = x.reshape(1,-1,x.shape[-1])
        x = x.transpose(1,2)
        conv3 = self.conv_1024_3(x)
        conv5 = self.conv_1024_5(x)
        conv7 = self.conv_1024_7(x)
        conv9 = self.conv_1024_9(x)
        x = torch.cat((x,conv3,conv7),1)
    
        x = x.transpose(1,2)
     
        x = self.mab(x,x)
        
        x = x.reshape(x.shape[0]*x.shape[1],-1)
        # x = self.fc1(x)
        x = self.SE(x)
        # x = self.fc2(x)
        if isTrain:
            # cos = self.maxmin_feature(x[:1024,:],x[1024:,:])
            x = self.classifier(x)
            return 0,x
        else:
            x = self.classifier(x)
            return x
    
    # snippet 维度卷积
    # def forward(self,x,isTrain):
        
    #     x = torch.unsqueeze(x, 1)
    #     x = self.mab(x,x)
    #     x = x.reshape(-1,32,x.shape[-1])
    #     x = x.transpose(1,2)
    #     x = self.conv(x)
    #     x = x.transpose(1,2)
    #     x = x.reshape(x.shape[0]*x.shape[1],-1)
    #     # x = self.fc1(x)
    #     # x = self.fc2(x)
    #     if isTrain:
    #         # cos = self.maxmin_feature(x[:1024,:],x[1024:,:])
    #         x = self.classifier(x)
    #         return 0,x
    #     else:
    #         x = self.classifier(x)
    #         return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Rnn(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=6, n_layer=2, n_classes=1):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.dense1 = nn.Linear(120, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        out, (h_n, c_n) = self.lstm(x)
        out = torch.transpose(out, 0, 1)
        x = temp.view(temp.shape[0],-1)
        # 此时可以从out中获得最终输出的状态h
        # x = out[:, -1, :]
        # x = h_n[-1, :, :]
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.classifier(x)
        return x

class RNN(nn.Module):
    def __init__(self,n_features):
        super(RNN, self).__init__()
        self.L = 32
        self.D = 32
        self.K = 1
        self.lstm = nn.LSTM(n_features, self.L, 2, batch_first=True)
        self.conv1 = nn.Conv1d(n_features, n_features, 3, padding=1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(0.6),
            
        )
        self.SE = SELayer(self.D)
        self.classifier = nn.Sequential(
            nn.Linear(self.D, 1),
            nn.Sigmoid()
        )
    

    def forward(self, x,isTrain):
        x = torch.transpose(x, 0, 1)
        out, (h_n, c_n) = self.lstm(x)
        out = torch.transpose(out, 0, 1)
        out = out.reshape(out.shape[0]*out.shape[1],-1)

        out = self.fc2(out)  # NxK
        # out = self.SE(out)
        Y_prob = self.classifier(out)

        return 0,Y_prob



class Model(nn.Module):
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs,isTrain):
        x = self.relu(self.fc1(inputs))
        x = self.dropout(x)
        # hidden = x
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        x = self.dropout(x)
        if isTrain:
            return 0,x
        else:
            return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,  = x.size()
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)
        # return x * y


if __name__ == '__main__':


    torch.manual_seed(seed=20200910)
    data_in = torch.randn(8,32,300,300)
    SE = SELayer(32) 
    data_out = SE(data_in)
    print(data_in.shape)  # torch.Size([8, 32, 300, 300])
    print(data_out.shape)  # torch.Size([8, 32, 300, 300])
    net = Rnn(6, 10, 2, 6)

class SEAttention(nn.Module):
    def __init__(self,n_features):
        super(SEAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        
        self.fc1 = nn.Sequential(
            nn.Linear(n_features, self.L),
            nn.ReLU(),
        )

        
        self.fc2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(self.D, self.D)
        )
        self.SE = SELayer(self.D)
        self.classifier = nn.Sequential(
            nn.Linear(self.D, 1),
            nn.Sigmoid()
        )
    
    def maxmin_feature(self,normal_features,abnor_features):
        normal_features = normal_features.reshape(-1,32,normal_features.shape[1])
        abnor_features = abnor_features.reshape(-1,32,abnor_features.shape[1])
        batch_sim = []
        for i in range(len(normal_features)):
            
            n_fea = normal_features[i]
            a_fea = abnor_features[i]
            a_max_index = torch.argsort(torch.sum(torch.square(n_fea),1),descending=True)[:8]
            n_max_index = torch.argsort(torch.sum(torch.square(a_fea),1),descending=True)[:16]
            n_select_fea = torch.sum(n_fea[n_max_index],0)
            a_select_fea = torch.sum(a_fea[a_max_index],0)
            
            batch_sim.append(F.cosine_similarity(n_select_fea, a_select_fea,0))
            
        return torch.tensor(batch_sim).mean()
    def forward(self, x,isTrain):
        
        H = self.fc1(x)  # NxL

        H = self.fc2(H)  # NxK
        H = self.SE(H)
        if isTrain:
            # cos = self.maxmin_feature(H[:1024,:], H[1024:,:])
            Y_prob = self.classifier(H)
            return 0,Y_prob

        Y_prob = self.classifier(H)

        return Y_prob

 



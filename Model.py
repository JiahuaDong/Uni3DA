from model_utils import *
from easydl import *
from torchvision import models
import pdb
import os
import torch.nn.functional as F
from pointnet2_cls_ssg import get_model

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y

# Grad Reversal
class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm

class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss().cuda()

    def get_metric_loss(self, x, ref):
        '''
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        '''

        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) * 64.  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return self.ce(score, gt_label)

    def forward(self, x, refs):
        loss = 0.
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss

# Generator
class Pointnet_g(nn.Module):
    def __init__(self):
        super(Pointnet_g, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)
        self.prediction_modules = nn.ModuleList()
        multi = 1
        n_rkhs = 512
        mid_channel = min(int(multi * 128), n_rkhs)
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(int(multi * 64), mid_channel, 1),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channel, n_rkhs, 1),
                Normalize(dim=1)
            )
        )

        # mid_channel = min(int(multi * 512), n_rkhs)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 512), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, n_rkhs, 1),
        #         Normalize(dim=1)
        #     )
        # )

        mid_channel = min(int(multi * 1024), n_rkhs)
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(int(multi * 1024), mid_channel, 1),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channel, n_rkhs, 1),
                Normalize(dim=1)
            )
        )

    def forward(self, x, node = False):
        lg_f = []
        out = []
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        x, node_fea, node_off = self.conv3(x, x_loc)  # x = [B, dim, num_node, 1]/[64, 64, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv5(x)

        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)
  
        x = self.bn1(x)
        lg_f.append(node_fea.squeeze(3))
        lg_f.append(x.unsqueeze(2))
        i = 0
        for prediction_modules in self.prediction_modules:
            out.append(prediction_modules(lg_f[i]))
            i = i + 1

        # x = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
        x = torch.cat((self.adaptive_maxpool(out[0].squeeze(2)), x.unsqueeze(2)), dim=1).squeeze(2)

        if node == True:
            return x, out, node_fea
        else:
            return x, node_fea


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        x = x.squeeze(-1)
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, in_dim, num_class=10, bottle_neck_dim=256):
        super(Pointnet_c, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, num_class)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))
    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

#discrimentor
class Discrimentor(nn.Module):
    def __init__(self, in_feature):
        super(Discrimentor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y
        
class Net_MDA(nn.Module):
    def __init__(self, model_name='Pointnet'):
        super(Net_MDA, self).__init__()
        if model_name == 'Pointnet':
            self.g = Pointnet_g()
            self.png = PointNetfeat(global_feat=True, feature_transform=True)
            self.png2 = get_model(num_class = 6,normal_channel=False)
            self.in_feature = 1536
            self.attention_s = CALayer(64*64)
            self.attention_t = CALayer(64*64)
            self.c1 = Pointnet_c(in_dim=self.in_feature)
            self.c2 = Pointnet_c(in_dim=self.in_feature)
            self.d = Discrimentor(self.in_feature)
            self.ds = Discrimentor(self.in_feature)
    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False, node_adaptation_t=False, discrimentor=False):
        x, feat_ori, node_idx = self.g(x, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis ==True:
            return node_idx

        # collect mid-level feat
        if mid_feat == True:
            return x, feat_ori
        
        if node_adaptation_s == True:
            # source domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s
        elif node_adaptation_t == True:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_t

        if adaptation == True:
            x = grad_reverse(x, constant)
        if discrimentor == True:
            y = self.d(x)
            return y

        y1 = self.c1(x)
        y2 = self.c2(x)
        return y1, y2

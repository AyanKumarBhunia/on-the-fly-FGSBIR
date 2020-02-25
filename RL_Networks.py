import torch
import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import math
from torch.distributions.multivariate_normal import MultivariateNormal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class backbone_network(nn.Module):
    def __init__(self):
        super(backbone_network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)
        #self.backbone.aux_logits = False
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        return x

    def fix_backbone(self):
        for name, x in self.named_parameters():
            x.requires_grad = False


class Policy(nn.Module):
    def __init__(self, state_dim = 2048, action_dim = 64, log_std=0):
        super(Policy, self).__init__()
        self.actor = nn.Linear(2048, 64)
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * log_std)

    def forward(self, x):
        action_mean = self.actor(x)
        return action_mean

    def fix_network(self):
        for name, x in self.named_parameters():
            if name in ['action_log_std']:
                x.requires_grad = False
                print(name, x.requires_grad)

    def select_action(self, x):
        action_mean = self.forward(x)
        m = torch.distributions.Normal(action_mean, torch.exp(0.5*self.action_log_std))
        sketch_anchor_embedding = m.sample()
        log_prob = m.log_prob(sketch_anchor_embedding).sum()
        entropy = m.entropy()
        return action_mean, sketch_anchor_embedding, log_prob, entropy


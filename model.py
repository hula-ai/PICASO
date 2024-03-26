#import torch
import torch.nn as nn
import timm
from set_modules import *

class net(nn.Module):
    def __init__(
        self,
        dim_input, #576,
        dim_hidden,
        num_heads,
        num_outputs=1,
        dim_output=1,
        ln=False,
    ):
        super(net, self).__init__()

        ##################################
        # Create the model
        ##################################
        self.prep = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=dim_input)

        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA_v2(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        B, N, C, H, W = X.shape
        X = X.view(B * N, C, H, W)
        prep = self.prep(X)
        prep = prep.view(B, N, -1)
        s = self.dec(prep)
        return s.squeeze(2)

class DeepSet(nn.Module):
    def __init__(
            self,
            pool,
            dim_input,
            dim_hidden,
            dim_output=1,
    ):
        super(DeepSet, self).__init__()

        #Setup network
        self.prep = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=dim_input)
        self.d_dim = dim_hidden
        self.x_dim = dim_input

        if pool == 'max':
            self.phi = nn.Sequential(
                PermEqui2_max(self.x_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui2_max(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui2_max(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
            )
        elif pool == 'max1':
            self.phi = nn.Sequential(
                PermEqui1_max(self.x_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui1_max(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui1_max(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
            )
        elif pool == 'mean':
            self.phi = nn.Sequential(
                PermEqui2_mean(self.x_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui2_mean(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui2_mean(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
            )
        elif pool == 'mean1':
            self.phi = nn.Sequential(
                PermEqui1_mean(self.x_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui1_mean(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
                PermEqui1_mean(self.d_dim, self.d_dim),
                nn.ELU(inplace=True),
            )

        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, dim_output),
        )

    def forward(self, X):
        B, N, C, H, W = X.shape
        X_flat = X.view(B * N, C, H, W)
        prep = self.prep(X_flat).view(B, N, -1)
        phi_output = self.phi(prep)
        #sum_output = phi_output.mean(1)
        ro_output = self.ro(phi_output)
        #H_enc = self.D(prep)
        return ro_output.squeeze(2)

class D(nn.Module):

  def __init__(self, d_dim, x_dim=2, pool = 'mean'):
    super(D, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.ELU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 4),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output = phi_output.mean(1)
    ro_output = self.ro(sum_output)
    return ro_output


class DTanh(nn.Module):

  def __init__(self, d_dim, x_dim=2, pool = 'mean'):
    super(DTanh, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 4),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm
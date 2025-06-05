import torch.nn as nn

from sfdm.models.utils.embedder import get_embedder
from sfdm.models.utils.siren import *

class DisplacementNetwork(nn.Module):
    def __init__(self, in_feature_dim, dims, d_in=3, d_out=1, multires=0, weight_norm=True,\
                skip_in=[], displace_feature_dim=128, use_feature=True \
                ):
        super().__init__()
        self.use_feature = use_feature
        if self.use_feature:
            dims = [d_in + in_feature_dim] + dims + [d_out + displace_feature_dim]
        else:
            dims = [d_in] + dims + [d_out + displace_feature_dim]
        self.skip_in = skip_in
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            if self.use_feature:
                dims[0] = input_ch + in_feature_dim
            else:
                dims[0] = input_ch
        else:
            input_ch = d_in

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                in_dim = dims[l] + input_ch
            else:
                in_dim = dims[l]
            lin = nn.Linear(in_dim, dims[l+1])
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x=None, feature=None):
        if len(x.shape) != len(feature.shape):
            feature = feature.squeeze(0)
        if x is not None:
            if self.embed_fn is not None:
                x = self.embed_fn(x)
            input_x = x
            if self.use_feature and (not feature is None):
                if x.shape[0] != feature.shape[0] and feature.shape[0] == 1:
                    feature = feature.repeat(x.shape[0],1)
                assert feature.shape[0] == x.shape[0], print('in correct net, shape_code.dim != x.dim', feature.shape, x.shape)
                input_x =  torch.cat([x, feature], dim=-1)
        else:
            input_x = feature
            
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                input_x = torch.cat([input_x, x],-1)
            input_x = lin(input_x)
            if l < self.num_layers - 2:
                input_x = self.relu(input_x)

        return input_x
    
    def update_embed_fn(self, alpha):
        if self.embed_fn is not None:
            self.embed_fn.update_alpha(alpha)

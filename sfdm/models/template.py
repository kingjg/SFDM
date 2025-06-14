import sys
import torch
import pdb
from torch import nn
import torch.nn.functional as F
from .utils.conditioned_nerual_fields import CONDITIONED_NEURAL_FIELD
from sfdm.models.utils import rigid
from sfdm.models.utils.siren import *
from sfdm.models.render_network import _xavier_init, _kaiming_init
from sfdm.models.utils.embedder2 import *


def warp(xyz, deformation, warp_type):
    """
    :param xyz: (B,N,3)
    :param deformation: (B,N,6)/(B,N,3)
    :param warp_type: 'translation' or 'se3'
    :return: warped xyz
    """
    if len(xyz.size()) == 2:
        xyz = xyz.unsqueeze(0)
    if len(deformation.size()) == 2:
        deformation = deformation.unsqueeze(0)

    B, N, _ = xyz.size()
    if warp_type == 'translation':
        return xyz + deformation
    elif warp_type == 'se3':
        w = deformation[:, :, :3]
        v = deformation[:, :, 3:6]
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = rigid.exp_se3(screw_axis, theta)
        warped_points = rigid.from_homogenous((transform @ (rigid.to_homogenous(xyz)[..., None])).squeeze(-1))
        return warped_points
    else:
        raise ValueError


class Embeddings(nn.Module):
    def __init__(self, id_num, exp_num, id_dim, exp_dim, initial_std=0.01, extend=True):
        super().__init__()
        # std=1/math.sqrt(self. id/exp _dim) or 0.01
        self.extend = extend
        self.id_embeddings = nn.Embedding(id_num, id_dim)
        self.register_buffer('valid_id', torch.zeros(id_num).long())
        nn.init.normal_(self.id_embeddings.weight, mean=0, std=initial_std)
        if extend:
            self.register_buffer('valid_exp', torch.zeros(id_num * exp_num).long())
            self.exp_embeddings = nn.Embedding(id_num * exp_num, exp_dim)
        else:
            self.register_buffer('valid_exp', torch.zeros(exp_num).long())
            self.exp_embeddings = nn.Embedding(exp_num, exp_dim)
        nn.init.normal_(self.exp_embeddings.weight, mean=0, std=initial_std)

    @property
    def mean(self, name):
        assert name in ['ID', 'EXP']
        if name == 'ID':
            return torch.mean(self.id_embeddings(self.valid_id), dim=0)
        elif name == 'EXP':
            return torch.mean(self.exp_embeddings(self.valid_exp), dim=0)
        else:
            raise ValueError

    def forward(self, exp_infor, id_infor):
        '''
            exp_infor: index(es) or given embedding(s) or None
            id_infor: index(es) or given embedding(s) or None
        '''
        if exp_infor is not None:
            assert isinstance(exp_infor, torch.Tensor)
            if exp_infor.dtype in (torch.int32, torch.int64):
                if self.extend:
                    assert id_infor.dtype in (torch.int32, torch.int64), \
                         'ID index must be provided when using extending embeddings.'
                    exp_idx = exp_infor * self.id_embeddings.num_embeddings + id_infor
                    exp_code = self.exp_embeddings(exp_idx.long())
                    self.valid_exp[exp_idx] = 1
                else:
                    exp_code = self.exp_embeddings(exp_infor.long())
                    self.valid_exp[exp_infor.long()] = 1
            else:
                exp_code = exp_infor
        else:
            assert id_infor is not None
            exp_code = None

        if id_infor is not None:
            assert isinstance(id_infor, torch.Tensor)
            if id_infor.dtype in (torch.int32, torch.int64):
                id_code = self.id_embeddings(id_infor.long())
                self.valid_id[id_infor.long()] = 1
            else:
                id_code = id_infor
        else:
            assert exp_infor is not None
            id_code = None

        if exp_code is None:
            return id_code
        elif id_code is None:
            return exp_code
        else:
            return exp_code, id_code


class LandmarkNets(nn.Module):
    def __init__(self,
                 exp_dim,
                 id_dim,
                 hidden_feature,
                 kpt_num,
                 template_kpts,
                 sparse_index=torch.LongTensor([36, 45, 30, 48, 54])):
        super().__init__()
        self.kpt_num = kpt_num
        self.register_buffer('sparse_index', sparse_index)
        self.register_buffer('template_kpts', template_kpts)
        self.exp_landmark_net = nn.Sequential(nn.Linear(id_dim + exp_dim, hidden_feature), nn.LeakyReLU(inplace=True),
                                              nn.Linear(hidden_feature, hidden_feature), nn.LeakyReLU(inplace=True),
                                              nn.Linear(hidden_feature, 3 * kpt_num))
        self.id_landmark_net = nn.Sequential(nn.Linear(id_dim, hidden_feature), nn.LeakyReLU(inplace=True),
                                             nn.Linear(hidden_feature, hidden_feature), nn.LeakyReLU(inplace=True),
                                             nn.Linear(hidden_feature, 3 * kpt_num))

    def get_template_kpts(self, sparse=False):
        return self.template_kpts[self.sparse_index, :] if sparse else self.template_kpts

    @property
    def get_sparse_index(self):
        return self.sparse_index

    def get_sparse_landmark(self, landmark):
        return landmark[:, self.sparse_index, :]

    def get_exp_landmark(self, exp_code, id_code):
        code = torch.cat((exp_code, id_code), dim=-1)
        return self.exp_landmark_net(code).view(-1, self.kpt_num, 3)

    def get_id_landmark(self, id_code):
        return self.id_landmark_net(id_code).view(-1, self.kpt_num, 3)

    def forward(self, **kwargs):
        raise NotImplementedError


class ImFaceSDFTemplate(nn.Module):
    def __init__(self, model_cfg, dataset_cfg):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.module_topology = ['embeddings', 'landmark_nets', 'exp_field', 'id_field', 'template_field']

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build(self):
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_embeddings(self):
        if self.model_cfg.get('EMBEDDINGS', None) is None:
            return None
        return Embeddings(self.dataset_cfg.ID_NUM, self.dataset_cfg.EXP_NUM, self.dataset_cfg.ID_DIM, self.dataset_cfg.EXP_DIM,
                          self.model_cfg.EMBEDDINGS.INITIAL_STD, self.model_cfg.EMBEDDINGS.EXTEND)

    def build_landmark_nets(self):
        if self.model_cfg.get('LANDMARK_NETS', None) is None:
            return None
        return LandmarkNets(self.dataset_cfg.EXP_DIM, self.dataset_cfg.ID_DIM, self.model_cfg.LANDMARK_NETS.HIDDEN_FEATURES,
                            len(self.dataset_cfg.TEMPLATE_KPTS), self.dataset_cfg.TEMPLATE_KPTS,
                            # self.dataset_cfg.LANDMARK_DIM, self.dataset_cfg.TEMPLATE_KPTS,
                            torch.LongTensor(self.dataset_cfg.SPARSE_LANDMARK_INDEX))

    def build_exp_field(self):
        if self.model_cfg.get('EXP_DEFORMATION_FIELD', None) is None:
            return None
        self.model_cfg.EXP_DEFORMATION_FIELD.OUT_DIM = 6 if self.model_cfg.WARP_TYPE == 'se3' else 3
        return CONDITIONED_NEURAL_FIELD[self.model_cfg.EXP_DEFORMATION_FIELD.NAME](self.model_cfg.EXP_DEFORMATION_FIELD,
                                                                                   self.dataset_cfg.EXP_DIM)

    def build_id_field(self):
        if self.model_cfg.get('ID_DEFORMATION_FIELD', None) is None:
            return None
        self.model_cfg.ID_DEFORMATION_FIELD.OUT_DIM = 7 if self.model_cfg.WARP_TYPE == 'se3' else 4
        return CONDITIONED_NEURAL_FIELD[self.model_cfg.ID_DEFORMATION_FIELD.NAME](self.model_cfg.ID_DEFORMATION_FIELD,
                                                                                  self.dataset_cfg.ID_DIM)

    def build_template_field(self):
        if self.model_cfg.get('TEMPLATE_FIELD', None) is None:
            return None
        return CONDITIONED_NEURAL_FIELD[self.model_cfg.TEMPLATE_FIELD.NAME](self.model_cfg.TEMPLATE_FIELD, None)

    def forward(self, **kwargs):
        raise NotImplementedError

    
class TempBRDF(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        low_rank = 3
        self.num_rough, self.num_coeff, self.num_specular, self.num_diffuse = 1, low_rank, 1, 3

        self.num_layers = len(dims)
        self.skip = 4

        net_list = []
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            if l == 0:
                net_list.append(ModulatedLayer(dims[l], out_dim, first_layer_sine_init))
            elif l == self.skip:
                net_list.append(ModulatedLayer(dims[l] + dims[0], out_dim, sine_init))
            elif l < self.num_layers - 2:
                net_list.append(ModulatedLayer(dims[l], out_dim, sine_init))
            else:
                self.final_layer = torch.nn.Linear(256, self.num_rough + self.num_specular + self.num_diffuse + self.num_coeff)
                _xavier_init(self.final_layer)
        self.template_spatial_network = nn.ModuleList(net_list)


    def forward(self, coords):
        points = torch.cat([coords], dim=-1)
        x = points
        for index, layer in enumerate(self.template_spatial_network):
            if index % self.skip == 0 and index > 0:
                x = torch.cat([x, points], dim=-1)
            x = layer(x)
        
        brdf_feature = x
        brdf_spatial = self.final_layer(x)

        return brdf_spatial, brdf_feature
    
class TempGradient(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        layers = []
        for i in range(4):
            if i == 0:
                dim_in = input_ch
                dim_out = 128
            else:
                dim_in = 128
                dim_out = 128
            if i == 0:
                layers.append(ModulatedLayer(dim_in, dim_out, first_layer_sine_init))
            else:
                layers.append(ModulatedLayer(dim_in, dim_out, sine_init))
        linear = torch.nn.Linear(128, 3)
        _xavier_init(linear)
        layers.append(linear)
        self.template_grad_net = torch.nn.ModuleList(layers)

    def forward(self, coords):
        g_x = torch.cat([coords], dim=-1)
        for index, layer in enumerate(self.template_grad_net):
            g_feat = g_x
            g_x = layer(g_x)

        return g_x, g_feat
    

class FacialTemplate(ImFaceSDFTemplate):
    def __init__(self, config, model_cfg, dataset_cfg):
        super().__init__(model_cfg=model_cfg, dataset_cfg=dataset_cfg)
        self.cfg = config
        self.register_buffer('template_kpts', self.dataset_cfg.TEMPLATE_KPTS)

        self.use_landmark_net = self.model_cfg.get('LANDMARK_NETS', None) is not None
        self.is_stage1 = self.cfg.get('is_stage1', False)
        self.use_temp_scatter = self.cfg.get('use_temp_scatter', False)

        self.neutral_idx = dataset_cfg.NUTRAL_EXP_INDEX
        self.training_losses = model_cfg.LOSS

        input_ch = 3
        self.embed_fn = None
        multires = self.cfg.get("template_multires", 0)
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn

        dims = [input_ch, 256, 256, 256, 256, 256, 256, 256, 256, 3]
        self.template_spatial_network = TempBRDF(dims)
        self.template_grad_network = TempGradient(input_ch)

        self.build()

    def forward(self, coords, exp_code, id_code, only_pts=False, need_deform=False):
        exp_input_dict = {'XYZ': coords, 'CODE': exp_code}
        if self.use_landmark_net:
            pred_exp_landmarks = self.landmark_nets.get_exp_landmark(exp_code, id_code)
            if only_pts:
                return pred_exp_landmarks
            exp_input_dict['KEY_POINTS'] = self.landmark_nets.get_sparse_landmark(pred_exp_landmarks)
        deformation_exp, deform_feat = self.exp_field(exp_input_dict)
        canonical_coords = warp(coords, deformation_exp, self.model_cfg.WARP_TYPE)

        id_input_dict = {'XYZ': canonical_coords, 'CODE': id_code}
        if self.use_landmark_net:
            pred_id_landmarks = self.landmark_nets.get_id_landmark(id_code)
            id_input_dict['KEY_POINTS'] = self.landmark_nets.get_sparse_landmark(pred_id_landmarks)
        id_output, id_feat = self.id_field(id_input_dict)
        deformation_id = id_output[:, :, :-1]
        correction = id_output[:, :, -1:]
        template_coords = warp(canonical_coords, deformation_id, self.model_cfg.WARP_TYPE)

        if self.embed_fn is not None:
            new_coords = self.embed_fn(template_coords)
        else:
            new_coords = template_coords

        brdf_spatial, brdf_feature = self.template_spatial_network(new_coords)
        g_x, g_feat = self.template_grad_network(new_coords)
        sc_x = torch.zeros([1, torch.cat([new_coords], dim=-1).shape[1], 21]).float().to(new_coords.device)

        temp_input_dict = {'XYZ': template_coords, 'CODE': None}
        if self.use_landmark_net:
            temp_input_dict['KEY_POINTS'] = self.landmark_nets.get_template_kpts(sparse=True)
        sdf_template, template_feat = self.template_field(temp_input_dict)

        if self.is_stage1:
            sdf = sdf_template
        else:
            sdf = sdf_template + correction

        geo_feature = torch.cat([id_feat, template_feat], -1)

        return {"sdf": sdf.squeeze(), "brdf": brdf_spatial, "scatter": sc_x, \
        "geo_feature": geo_feature, "template_coords": template_coords, \
        "albedo_grad": g_x, "brdf_feature": brdf_feature}

    def gradient(self, x, exp, id, device):
        x.requires_grad_(True)
        y = self.forward(x, exp, id)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

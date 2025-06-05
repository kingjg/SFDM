import torch
import torch.nn as nn
from sfdm.models.utils.film import *
from sfdm.models.utils.siren import *
from sfdm.models.utils.embedder2 import *
from sfdm.models.density import LaplaceDensity
import e3nn.o3 as o3


def _xavier_init(linear):
    torch.nn.init.xavier_uniform_(linear.weight.data)

def _kaiming_init(linear):
    torch.nn.init.kaiming_uniform_(linear.weight.data)


class RefNetwork(nn.Module):
    def __init__(
            self,
            config,
            d_in,
            d_out,
            dims,
            low_rank,
            brdf_network,
            calibration_network
    ):
        super().__init__()
        
        self.hidden_dim = dims[0]
        dims = list(dims)
        dims = [d_in] + dims + [d_out]
        self.skip = 4
        self.density = LaplaceDensity('cuda', params_init={'beta': 0.01}, beta_min=0.0001)
        self.num_layers = len(dims)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # dimension
        self.num_rough, self.num_coeff, self.num_specular, self.num_diffuse = 1, low_rank, 1, 3
        self.num_radius, self.num_scatter = 3, 18

        net_list = []

        self.cfg = config
        # self.use_displacement = config.use_displacement
        self.use_scatter = config.get('use_scatter', True)
        self.is_stage1 = config.get('is_stage1', False)
        self.use_feature = config.get('use_feature', True)
        self.use_siren = config.get('use_siren', True)
        self.edit_spec = config.get('edit_spec', False)

        # [spec, albedo, roughness]
        init = torch.FloatTensor([0.9, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.7])

        if self.is_stage1:
            self.brdf_offset_weight = 0.01
        else:
            self.temp_brdf_weight = nn.Parameter(init, requires_grad=True)
            self.scatter_weight = nn.Parameter(torch.FloatTensor([0.15]), requires_grad=True)

        input_ch = 3
        self.embed_fn = None
        multires = config.get('multires', 0)
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
        dims[0] = input_ch

        if not self.is_stage1:
            self.num_layers = len(dims)
            if self.use_feature:
                dims[0] = input_ch
                self.sdf_feat_layer = 2
            else:
                dims[0] = input_ch
        else: 
            self.num_layers = len(dims)
            if self.use_feature:
                dims[0] = input_ch
                self.sdf_feat_layer = 0
            else:
                dims[0] = input_ch

        self.g_layer = 4
        self.brdf_feature_layer = 6

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            in_dim = dims[l]
            if self.use_feature and l == self.sdf_feat_layer:
                in_dim += 64 + 64 + 64
            if l == self.brdf_feature_layer:
                in_dim += 128
            if l == self.g_layer:
                in_dim += 3
            if l == 0 and self.is_stage1:
                in_dim += 128

            if l == 0:
                if not self.use_siren:
                    linear = nn.Linear(in_dim, out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(in_dim, out_dim, first_layer_sine_init))
            elif l == self.skip:
                if not self.use_siren:
                    linear = nn.Linear(in_dim + dims[0], out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(in_dim + dims[0], out_dim, sine_init))
            elif l < self.num_layers - 2:
                if not self.use_siren:
                    linear = nn.Linear(in_dim, out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(in_dim, out_dim, sine_init))
            else:
                self.final_layer = torch.nn.Linear(256, self.num_rough + self.num_specular + self.num_diffuse + self.num_coeff)
                _xavier_init(self.final_layer)

        if not self.is_stage1:
            self.ofs_st2 = nn.ModuleList(net_list)
        else:
            self.ofs_st1 = nn.ModuleList(net_list)
        
        net_list = []
        if (not self.is_stage1):
            dims = [input_ch, 256, 256, 256, 256, 256, 256, 256, self.num_scatter + self.num_radius, 0]
        else:
            dims = [input_ch + 128, 256, 256, 256, self.num_scatter + self.num_radius, 0]
        
        self.skip_scatter = 4
        self.num_layers_scatter = len(dims)
        for l in range(0, self.num_layers_scatter - 1):
            out_dim = dims[l + 1]
            if l == 0:
                if not self.use_siren:
                    linear = nn.Linear(dims[l], out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(dims[l], out_dim, first_layer_sine_init))
            elif l == self.skip_scatter:
                if not self.use_siren:
                    linear = nn.Linear(dims[l] +  dims[0], out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(dims[l] + dims[0], out_dim, sine_init))
            elif l < self.num_layers_scatter - 2:
                if not self.use_siren:
                    linear = nn.Linear(dims[l], out_dim)
                    net_list.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
                else:
                    net_list.append(ModulatedLayer(dims[l], out_dim, sine_init))

        if not self.is_stage1:
            self.scatter_params_network2 = nn.ModuleList(net_list)

        if not self.is_stage1:
            net_list = []
            light_dim = 128
            dims = [input_ch + self.num_radius + light_dim, 256, 256, 256, 256, 3]
            self.num_layers_scatter = len(dims)
            for l in range(0, self.num_layers_scatter - 1):
                out_dim = dims[l + 1]
                if l < self.num_layers_scatter - 2:
                    net_list.append(ModulatedLayer(dims[l], out_dim, sine_init))
                else:
                    self.final_scatter_light_layer = torch.nn.Linear(256, 3)
                    _xavier_init(self.final_scatter_light_layer)
            self.scatter_light_network = nn.ModuleList(net_list)

            self.light_compress = nn.Linear(121 * 3, light_dim)

            net_list = []
            dims = [input_ch + self.num_scatter, 256, 256, 256, 256, 3]
            self.num_layers_scatter = len(dims)
            for l in range(0, self.num_layers_scatter - 1):
                out_dim = dims[l + 1]
                if l < self.num_layers_scatter - 2:
                    net_list.append(ModulatedLayer(dims[l], out_dim, sine_init))
                else:
                    self.final_scatter_material_layer = torch.nn.Linear(256, 3)
                    _xavier_init(self.final_scatter_material_layer)
            self.scatter_material_network = nn.ModuleList(net_list)

        # PBR Module
        layer_num = 4
        layer_dim = 128
        layers = []
        for i in range(layer_num):
            if i == 0:
                dim_in = 2
                dim_out = layer_dim
            else:
                dim_in = layer_dim
                dim_out = layer_dim
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.B_network = torch.nn.ModuleList(layers)
        self.B_basis = torch.nn.Linear(layer_dim, self.num_coeff * 2)
        _xavier_init(self.B_basis)

        layer_num = 4
        layer_dim = 128
        layers = []
        for i in range(layer_num):
            if i == 0:
                dim_in = 6
                dim_out = layer_dim
            else:
                dim_in = layer_dim
                dim_out = layer_dim
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.F_network = torch.nn.ModuleList(layers)
        self.F_basis = torch.nn.Linear(layer_dim, 1)
        # nn.init.xavier_uniform_(self.F_basis.weight, gain=0.01)
        nn.init.constant_(self.F_basis.bias, -5)

        # calibration network
        layers = []
        net_depth_condition = calibration_network.depth
        net_width_condition = calibration_network.hidden_dim
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = 50
                dim_out = net_width_condition
            elif i == net_depth_condition - 1:
                dim_in = net_width_condition
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        linear = torch.nn.Linear(net_width_condition, 9)
        _xavier_init(linear)
        layers.append(linear)
        self.calibration_network = torch.nn.Sequential(*layers)


    def forward(self, points, view_dirs, normals=None, calibration_code=None, sh_light=None, \
                brdf_latent=None, temp_brdf=None, temp_scatter=None, sdf_feature=None, brdf_feature=None, albedo_g=None):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        input = torch.cat([points], dim=-1)

        if not self.is_stage1:
            x = input
            for index, layer in enumerate(self.ofs_st2):
                if index % self.skip == 0 and index > 0:
                    x = torch.cat([x, input], dim=-1)
                if self.use_feature and index == self.sdf_feat_layer:
                    x = torch.cat([x, sdf_feature[0]], dim=-1)
                if index == self.brdf_feature_layer:
                    x = torch.cat([x, brdf_feature[0]], dim=-1)
                if index == self.g_layer:
                    x = torch.cat([x, albedo_g[0]], dim=-1)
                x = layer(x)
        else:
            # input = torch.cat([input], dim=-1)
            x = torch.cat([input], dim=-1)
            for index, layer in enumerate(self.ofs_st1):
                if index == 0:
                    x = torch.cat([x, brdf_latent.repeat(input.shape[0], 1)], dim=-1)
                if self.use_feature and index == self.sdf_feat_layer:
                    x = torch.cat([x, sdf_feature[0]], dim=-1)
                if index % self.skip == 0 and index > 0:
                    x = torch.cat([x, input], dim=-1)
                x = layer(x)

        out_brdf_feature = x.clone()
        y = self.final_layer(x)

        if not self.is_stage1:
            sc_x = input
            for index, layer in enumerate(self.scatter_params_network2):
                if index % self.skip_scatter == 0 and index > 0:
                    sc_x = torch.cat([sc_x, input], dim=-1)
                sc_x = layer(sc_x)

        if not temp_brdf is None:
            brdf_offset = y.clone()
            if self.is_stage1:
                y = brdf_offset * self.brdf_offset_weight + (1 - self.brdf_offset_weight) * temp_brdf[0]
            else:
                y = temp_brdf[0] * self.temp_brdf_weight + brdf_offset
        
        normals = normals.squeeze(0)

        if not self.is_stage1:
            scale = self.sigmoid(sc_x[:, :self.num_radius]).view(-1, self.num_radius) 
            sc_params = sc_x[:, self.num_radius:]
            sc_params = self.sigmoid(sc_params)
 
            z_l = self.relu(self.light_compress(sh_light.detach().view(1, -1)))
            sc_x = torch.cat([points, scale, z_l.repeat(sc_x.shape[0], 1)], dim=1)
            for index, layer in enumerate(self.scatter_light_network):
                sc_x = layer(sc_x)
            sc_light_y = self.final_scatter_light_layer(sc_x)

            sc_x = torch.cat([points, sc_params], dim=1)
            for index, layer in enumerate(self.scatter_material_network):
                sc_x = layer(sc_x)
            sc_material_y = self.final_scatter_material_layer(sc_x)

        roughness = self.softplus(y[..., 0:1])
        basis_coeff = y[..., 1:self.num_coeff + 1]
        albedo = self.sigmoid(y[..., self.num_coeff + 1:self.num_coeff + 4])

        # spec editing
        spec = self.sigmoid(y[..., self.num_coeff + 4:]) * self.cfg.get('spec_rate', 1)
        if self.edit_spec:
            spec = self.sigmoid(y[..., self.num_coeff + 4:]) * 1.5

        irradiance = e3_SH(lmax=10, directions=normals, sh=sh_light, lambert_coeff=True)
        irradiance = torch.relu(irradiance)
        diffuse = albedo * irradiance

        if self.use_scatter:
            diffuse_offset = self.sigmoid(sc_light_y) * self.sigmoid(sc_material_y)
            diffuse = diffuse_offset * self.scatter_weight + diffuse
        else:
            diffuse_offset = torch.zeros_like(points).float().to(diffuse.device)

        w0 = -view_dirs
        dot = normals * w0
        dot = dot.sum(dim=-1, keepdim=True)
        wr = 2 * normals * dot - w0
        light_transport = e3_SH(lmax=10, directions=wr, sh=sh_light, rho=roughness)
        light_transport = torch.relu(light_transport)

        x = torch.cat([dot, roughness], dim=-1)
        for index, layer in enumerate(self.B_network):
            x = layer(x)
        B = self.B_basis(x)

        f0 = torch.cat([w0, normals], dim=-1)
        for index, layer in enumerate(self.F_network):
            f0 = layer(f0)
        f0 = self.sigmoid(self.F_basis(f0))

        f0 = (1 - dot) ** 5
        brdf = f0 * B[:, :3] + B[:, 3:]

        cs = brdf * basis_coeff
        cs = cs.sum(dim=-1, keepdim=True)

        spec_energy = spec * light_transport * self.sigmoid(cs)
        diffuse_energy = diffuse
        raw_rgb = diffuse + spec * light_transport * self.sigmoid(cs)
        albedo_energy = albedo

        exposure = self.calibration_network(calibration_code)
        affine = exposure[:, :9].reshape(3, 3)

        raw_rgb = torch.matmul(raw_rgb, affine)
        raw_rgb = torch.clamp(raw_rgb, 0.0, 1.0)
        diff_ = torch.matmul(diffuse_energy, affine)
        diff_ = torch.clamp(diff_, 0.0, 1.0)
        spec_ = torch.matmul(spec_energy, affine)
        spec_ = torch.clamp(spec_, 0.0, 1.0)
        albedo = torch.matmul(albedo_energy * 3.1415926, affine)
        albedo = torch.clamp(albedo, 0.0, 1.0)

        roughness = torch.clamp(roughness, 0.0, 1.0)
        roughness = roughness.repeat(1, 3)

        light_reg = torch.cat([light_transport, irradiance], dim=0)
        
        output = {
            'raw_rgb': raw_rgb,
            'diffuse': diff_,
            'albedo': albedo,
            'spec': spec_,
            'spec_energy': spec_energy,
            'albedo_energy': albedo_energy,
            'diffuse_energy': diffuse_energy,
            'light_reg': light_reg,
            'brdf_offset': brdf_offset,
            'roughness': roughness,
            'diffuse_offset': diffuse_offset,
            'out_brdf_feature': out_brdf_feature,
        }

        return output


lambert_sh_k = [3.1415926535897927,
                2.0943951023931957,
                0.7853981633974483,
                0.0,
                -0.13089969389957473,
                0.0,
                0.04908738521234052,
                0.0,
                -0.024543692606170262,
                0.0,
                0.014317154020265985]


def e3_SH(lmax, directions, sh=None, rho=None, lambert_coeff=False):
    d = directions[..., [1, 2, 0]]
    basis = torch.zeros((directions.shape[0], (lmax + 1) ** 2)).to(directions.device).float()
    for i in range(lmax + 1):
        if lambert_coeff == True:
            basis[..., i ** 2: (i + 1) ** 2] = o3.spherical_harmonics(i, d, normalize=False) * lambert_sh_k[i]
        else:
            basis[..., i ** 2: (i + 1) ** 2] = o3.spherical_harmonics(i, d, normalize=False) * torch.exp(
                - i * (i + 1) / 2 * rho)
    basis = basis.unsqueeze(-1)
    tmp = basis * sh  # (N, 25 or 9, 3)
    return tmp.sum(-2)  # (N, 3)

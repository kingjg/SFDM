import torch
from torch import nn
from torch.nn import functional as F


class IDRLoss(nn.Module):
    def __init__(self, device, config, rgb_weight, eikonal_weight, embedding_weight, spec_weight, light_reg, residual_weight, brdf_offset_weight, \
                 diffuse_offset_weight=0, albedo_g_weight=0):
        super().__init__()
        self.device = device
        self.config = config
        self.rgb_weight = rgb_weight
        self.eikonal_weight = eikonal_weight
        self.embedding_weight = embedding_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.spec_weight = spec_weight
        self.light_reg = light_reg
        self.residual_weight = residual_weight
        self.brdf_offset_weight = brdf_offset_weight
        self.diffuse_offset_weight = diffuse_offset_weight
        self.albedo_g_weight = albedo_g_weight

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.squeeze()
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    def get_albedo_gradient_loss(self, rgb_values, rgb_gt):
        rgb_values = rgb_values.squeeze()
        rgb_gt = rgb_gt.squeeze()
        rgb_loss = self.l1_loss(rgb_values.sum(-1), rgb_gt.sum(-1))
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).to(self.device).float()
        eikonal_loss = ((grad_theta.norm(2, dim=2) - 1) ** 2).mean()
        return eikonal_loss

    def l_reg(self, sh):
        if sh == None:
            return torch.tensor(0.0).to(self.device).float()
        white = (sh[..., 0:1] + sh[..., 1:2] + sh[..., 2:3]) / 3.0
        return torch.mean(torch.abs(sh - white))

    def energy_loss(self, spec):
        if spec == None:
            return torch.tensor(0.0).to(self.device).float()
        return torch.mean(spec)
        #return torch.max(spec)

    def residual_loss(self, residual):
        if residual == None:
            return torch.tensor(0.0).to(self.device).float()
        l = torch.abs(residual)
        return torch.mean(l)

    def brdf_residual_loss(self, residual):
        if residual == None:
            return torch.tensor(0.0).to(self.device).float()
        l = torch.abs(residual)
        l = torch.mean(l, dim=0)
        # different variance of albedo and specular
        l[4:7] *= 0.0
        l[7:] *= 2.0
        return torch.mean(l)
    
    def scatter_residual_loss(self, residual):
        if residual == None:
            return torch.tensor(0.0).to(self.device).float()
        l = torch.abs(residual)
        l = torch.mean(l, dim=0)
        return torch.mean(l)

    def forward(self, model_outputs, ground_truth, id_latent, exp_latent, brdf_latent):
        rgb_gt = ground_truth['rgb'].to(self.device)
        spec_energy = model_outputs['spec_energy']
        sh_light = model_outputs['light_reg']
        residual_sdf = model_outputs['residual_sdf']
        albedo_g = model_outputs['albedo_gradient']
        pred_albedo_g = model_outputs['pred_albedo_gradient']

        if self.config.get("is_stage1", False):
            albedo_g_loss = self.get_albedo_gradient_loss(pred_albedo_g, albedo_g.detach()) * self.albedo_g_weight
        else:
            albedo_g_loss = 0.

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt) * self.rgb_weight     
        eikonal_loss = self.eikonal_weight * self.get_eikonal_loss(model_outputs['grad_theta'])
        id_loss = torch.mean((id_latent) ** 2) * self.embedding_weight
        exp_loss = torch.mean((exp_latent) ** 2) * self.embedding_weight # removed exp loss
        brdf_latent_loss = torch.mean((brdf_latent) ** 2) * self.embedding_weight
        energy_loss = self.energy_loss(spec_energy) * self.spec_weight
        lgt_loss = self.l_reg(sh_light) * self.light_reg
        residual_loss = self.residual_loss(residual_sdf) * self.residual_weight
        brdf_offset_loss = self.brdf_residual_loss(model_outputs['brdf_offset']) * self.brdf_offset_weight

        diffuse_offset_loss = torch.mean(torch.abs(model_outputs['diffuse_offset'])) * self.diffuse_offset_weight

        loss = rgb_loss + id_loss + exp_loss + eikonal_loss + lgt_loss + energy_loss \
                + residual_loss + brdf_latent_loss + brdf_offset_loss \
                + diffuse_offset_loss + albedo_g_loss
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'id_loss': id_loss,
            'exp_loss': exp_loss,
            'eikonal_loss': eikonal_loss,
            'lgt_loss': lgt_loss,
            'energy_loss': energy_loss,
            'residual_loss': residual_loss,
            'brdf_offset_loss': brdf_offset_loss,
            'brdf_latent_loss': brdf_latent_loss,
            'albedo_g_loss': albedo_g_loss,
        }

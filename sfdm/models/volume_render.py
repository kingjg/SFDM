import torch
import pdb
import torch.nn as nn
from sfdm.utils import rend_util
import torch.nn.functional as F


class Render(nn.Module):
    def __init__(self, device, config, decoder, ray_tracer, sample_network, render_network, residual):
        super().__init__()
        self.device = device
        self.config = config
        self.decoder = decoder
        self.ray_tracer = ray_tracer
        self.sample_network = sample_network
        self.rendering_network = render_network
        self.residual = residual
        self.is_stage1 = self.config.get('is_stage1', False)
        
        if not self.is_stage1:
            self.compress_sdf_feat = torch.nn.Sequential(nn.Linear(256, 128), torch.nn.ReLU(True))
            self.compress_brdf_feat = torch.nn.Sequential(nn.Linear(256, 128), torch.nn.ReLU(True))
    

    def gradient(self, x, exp, id):
        x.requires_grad_(True)
        decoder_output = self.decoder(x, exp, id)
        if not self.is_stage1:
            cat_feature = torch.cat([self.compress_sdf_feat(decoder_output['geo_feature']), self.compress_brdf_feat(decoder_output['brdf_feature']), decoder_output['albedo_grad']], -1)
            y = decoder_output['sdf'] + self.residual(x.squeeze(0), cat_feature)[:, 0]
        else:
            y = decoder_output['sdf']
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def forward(self, input):
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        self.id_latent = input["id_latent"]
        self.exp_latent = input["exp_latent"]
        self.brdf_latent = input["brdf_latent"]

        self.calibration_code = input["calibration_code"]
        self.sh_light = input["sh_light"]

        ray_dirs, cam_loc = rend_util.get_camera_params2(uv, pose, intrinsics)
        self.decoder.eval()

        def get_sdf(x, y, z):
            decoder_output = self.decoder(x.unsqueeze(0), z, y)
            if x.shape[0] == 1:
                decoder_output['geo_feature'] = decoder_output['geo_feature'].view(1, -1)
                decoder_output['brdf_feature'] = decoder_output['brdf_feature'].view(1, -1)
                decoder_output['albedo_grad'] = decoder_output['albedo_grad'].view(1, -1)
            cat_feature = torch.cat([self.compress_sdf_feat(decoder_output['geo_feature']), self.compress_brdf_feat(decoder_output['brdf_feature']), decoder_output['albedo_grad']], -1)
            return decoder_output['sdf'] + self.residual(x, cat_feature)[:, 0]
        
        if not self.is_stage1:
            self.residual.eval()
            with torch.no_grad():
                points, network_object_mask, dists = self.ray_tracer(
                    sdf=get_sdf,
                    cam_loc=cam_loc,
                    object_mask=input["object_mask"],
                    ray_directions=ray_dirs,
                    id_latent=self.id_latent,
                    exp_latent=self.exp_latent)
            self.residual.train()
        else:
            with torch.no_grad():
                points, network_object_mask, dists = self.ray_tracer(
                    sdf=lambda x, y, z: self.decoder(x.unsqueeze(0), z, y)['sdf'], # + self.residual(x),
                    cam_loc=cam_loc,
                    object_mask=input["object_mask"],
                    ray_directions=ray_dirs,
                    id_latent=self.id_latent,
                    exp_latent=self.exp_latent)

        self.decoder.train()

        g = self.gradient(points.unsqueeze(0), self.exp_latent, self.id_latent)

        decoder_output = self.decoder(points.unsqueeze(0), self.exp_latent, self.id_latent)
        if not self.is_stage1:
            cat_feature = torch.cat([self.compress_sdf_feat(decoder_output['geo_feature']), self.compress_brdf_feat(decoder_output['brdf_feature']), decoder_output['albedo_grad']], -1)
            surface_sdf = decoder_output['sdf'] + self.residual(points, cat_feature)[:, 0]
        else:
            surface_sdf = decoder_output['sdf']
        s_normals = g[:, 0, :]
        s_normals = s_normals.reshape(-1, 3)

        rgb_values = torch.zeros_like(points).float().to(self.device)
        diffuse_values = torch.zeros_like(points).float().to(self.device)
        albedo_values = torch.zeros_like(points).float().to(self.device)
        spec_values = torch.zeros_like(points).float().to(self.device)
        grad_theta = torch.tensor([[[0, 0, 0]]]).float().to(self.device)
        diffuse_energy = None
        spec_energy = None
        albedo_energy = None
        light_reg = None
        residual_sdf = None

        diffuse_offset = torch.zeros_like(points).float().to(self.device)

        out_mask = ~network_object_mask
        ray_dirs_out = ray_dirs
        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(self.device, cam_loc, ray_dirs_out,
                                                                                 r=1.0)
        mask_intersect = mask_intersect.squeeze(0)
        sphere_intersections = sphere_intersections.squeeze(0)
        out_mask = out_mask & mask_intersect
        sphere_intersections = sphere_intersections[out_mask]  # (N, 2)
        ray_dirs_out_mask = ray_dirs_out.reshape(-1, 3)[out_mask]  # (N. 3)

        # volume rendering for mask
        surface_mask = network_object_mask
        brdf_offset = None
        roughness_values = torch.zeros_like(points).float().to(self.device)
        albedo_g = None
        pred_albedo_g = None

        if surface_mask.sum() > 0 or sphere_intersections.sum() > 0:
            steps = torch.linspace(0, 1, steps=32).to(self.device)
            inter_dists = dists[surface_mask]
            near_surface = inter_dists - 0.005
            far_surface = inter_dists + 0.005
            near_surface = near_surface.unsqueeze(1)
            far_surface = far_surface.unsqueeze(1)
            near_out = sphere_intersections[:, :1]
            far_out = sphere_intersections[:, 1:]
            near = torch.cat([near_surface, near_out], dim=0)
            far = torch.cat([far_surface, far_out], dim=0)
            final_dists = near * (1. - steps) + far * (steps)  # (N, 32)
            final_dists = final_dists.reshape(-1, 1)  # (N * 32, 1)

            ray_dirs_out_surface = ray_dirs_out_mask.repeat(1, 32).reshape(-1, 3)  # (N * 32, 3)
            ray_dirs_surface = ray_dirs.reshape(-1, 3)[surface_mask]  # (N, 3)
            ray_dirs_surface = ray_dirs_surface.repeat(1, 32).reshape(-1, 3)  # (N * 32, 3)
            ray_dirs_all = torch.cat([ray_dirs_surface, ray_dirs_out_surface], dim=0)
            all_points = cam_loc + final_dists * ray_dirs_all  # (N * 32, 3)

            g = self.gradient(all_points.unsqueeze(0), self.exp_latent, self.id_latent)
            normals = g[:, 0, :]
            grad_theta = g[:, 0, :]
            normals = normals / torch.norm(normals, dim=2, keepdim=True)

            all_points.requires_grad_(True)

            decoder_output = self.decoder(all_points.unsqueeze(0), self.exp_latent, self.id_latent)
            if not self.is_stage1:
                decoder_output['geo_feature'] = self.compress_sdf_feat(decoder_output['geo_feature'])
                decoder_output['brdf_feature'] = self.compress_brdf_feat(decoder_output['brdf_feature'])
                cat_feature = torch.cat([decoder_output['geo_feature'], decoder_output['brdf_feature'], decoder_output['albedo_grad']], -1)

                residual_with_feature = self.residual(all_points, cat_feature)
                residual_sdf = residual_with_feature[:, 0]
                residual_feature = residual_with_feature[:, 1:].unsqueeze(0)
                sdf_out = decoder_output['sdf'] + residual_sdf 
                sdf_feature = torch.cat([decoder_output['geo_feature'], residual_feature], -1)
            else:
                sdf_out = decoder_output['sdf']
                sdf_feature = decoder_output['geo_feature']

            out = self.rendering_network(all_points, ray_dirs_all, normals=normals.squeeze(0),
                                         calibration_code=self.calibration_code, sh_light=self.sh_light, 
                                         brdf_latent=self.brdf_latent, temp_brdf=decoder_output['brdf'],
                                         temp_scatter=decoder_output['scatter'], sdf_feature=sdf_feature,
                                         brdf_feature=decoder_output['brdf_feature'], albedo_g=decoder_output['albedo_grad'])
            y = out['albedo_energy']
        
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            if self.is_stage1:
                albedo_g = torch.autograd.grad(
                    outputs=out['albedo_energy'],
                    inputs=all_points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            else:
                albedo_g = None
            pred_albedo_g = decoder_output['albedo_grad']

            sigma = self.rendering_network.density(sdf_out)

            brdf_offset = out['brdf_offset']
            diffuse_offset = out['diffuse_offset']

            diffuse_energy = out['diffuse_energy']
            spec_energy = out['spec_energy']
            albedo_energy = out['albedo_energy']
            light_reg = out['light_reg']

            surface_num = surface_mask.sum()

            rgb = out['raw_rgb']
            rgb = rgb.reshape(-1, 32, 3)
            weights = self.volume_rendering(final_dists.reshape(-1, 32), sigma)
            rgb_mask = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            rgb_values[surface_mask] = rgb_mask[:surface_num, :]
            rgb_values[out_mask] = rgb_mask[surface_num:, :]

            rgb = out['diffuse']
            rgb = rgb.reshape(-1, 32, 3)
            rgb_mask = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            diffuse_values[surface_mask] = rgb_mask[:surface_num, :]
            diffuse_values[out_mask] = rgb_mask[surface_num:, :]

            rgb = out['albedo']
            rgb = rgb.reshape(-1, 32, 3)
            rgb_mask = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            albedo_values[surface_mask] = rgb_mask[:surface_num, :]
            albedo_values[out_mask] = rgb_mask[surface_num:, :]

            rgb = out['spec']
            rgb = rgb.reshape(-1, 32, 3)
            rgb_mask = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            spec_values[surface_mask] = rgb_mask[:surface_num, :]
            spec_values[out_mask] = rgb_mask[surface_num:, :]

            rgb = out['roughness']
            rgb = rgb.reshape(-1, 32, 3)
            rgb_mask = torch.sum(weights.unsqueeze(-1) * rgb, 1)
            roughness_values[surface_mask] = rgb_mask[:surface_num, :]
            roughness_values[out_mask] = rgb_mask[surface_num:, :]

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'diffuse_values': diffuse_values,
            'albedo_values': albedo_values,
            'spec_values': spec_values,
            'normals': s_normals,
            'sdf_output': surface_sdf,
            'diffuse_energy': diffuse_energy,
            'albedo_energy': albedo_energy,
            'spec_energy': spec_energy,
            'light_reg': light_reg,
            'residual_sdf': residual_sdf,
            'brdf_offset': brdf_offset,
            'roughness_values': roughness_values,
            'diffuse_offset': diffuse_offset,
            'albedo_gradient': albedo_g,
            'pred_albedo_gradient': pred_albedo_g,
        }
        return output

    def volume_rendering(self, z_vals, density):
        density = density.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).to(self.device).unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).to(self.device), free_energy[:, :-1]],
                                        dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights
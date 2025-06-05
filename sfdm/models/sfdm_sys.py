import torch
from pytorch_lightning import LightningModule
import os

from sfdm.models.render_network import RefNetwork
from sfdm.models.utils.ray_tracing import RayTracing
from sfdm.models.loss import IDRLoss
from sfdm.models.utils.sample_network import SampleNetwork
from sfdm.models.volume_render import Render
from sfdm.models import template
from sfdm.models.network import DisplacementNetwork
from pyhocon.config_parser import ConfigFactory, ConfigTree
from sfdm.dataset.facescape_multi_dataset import FaceScape
from torch.utils.data import DataLoader
import sfdm.utils.general as utils
import torch.nn as nn
from sfdm.models.utils.embedder import get_embedder

class SFDM(LightningModule):
    def __init__(self, config, train_config, id2idx, exp2idx):
        super(SFDM, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.batch_size = self.config.dataset.batch_size
        self.is_stage1 = self.config.get('is_stage1', False)

        self.temp_feature_dim = 64
        self.deform_feature_dim = 64
        self.white_bkgd = True
        self.scene_bounding_sphere = 3.0
        self.displacement_feature_dim = 64

        # build template network
        self.decoder = template.FacialTemplate(config, train_config.MODEL, train_config.DATA_CONFIG)

        reload_path = config.load_path
        envs = torch.load(os.path.join(reload_path, 'pth', 'envs_last.pth'), map_location=torch.device('cpu'))
        self.decoder.load_state_dict(envs['model'], strict=False)

        # build rendering network
        self.rendering_network = RefNetwork(config, **config.rendering_network)
        device = 'cuda'
        self.ray_tracer = RayTracing(device, **config.ray_tracing)
        
        # build displacement network
        if not self.is_stage1:  
            in_dim = self.temp_feature_dim + self.deform_feature_dim + 128 + 3
            self.residual = DisplacementNetwork(in_feature_dim=in_dim, dims=[256, 256, 256, 256], d_in=3, d_out=1, multires=8, weight_norm=True,\
                    skip_in=[], displace_feature_dim=64, use_feature=config.get('residual_use_feature', True))

            self.sample_network = SampleNetwork()
            self.model = Render(device, config, self.decoder, self.ray_tracer, self.sample_network, self.rendering_network,
                                self.residual)
        else:
            self.sample_network = SampleNetwork()
            self.model = Render(device, config, self.decoder, self.ray_tracer, self.sample_network, self.rendering_network)
        
        # maximum number of subjects
        self.num_id = self.num_exp = 20

        self.id_lat_vecs = torch.nn.Embedding(num_embeddings=self.num_id, embedding_dim=128, max_norm=1.0)
        self.exp_lat_vecs = torch.nn.Embedding(num_embeddings=self.num_exp, embedding_dim=128, max_norm=1.0)
        self.brdf_lat_vecs = torch.nn.Embedding(num_embeddings=self.num_id, embedding_dim=128, max_norm=1.0)
        
        torch.nn.init.normal_(
            self.id_lat_vecs.weight.data,
            0.0,
            0.01,
        )
        torch.nn.init.normal_(
            self.exp_lat_vecs.weight.data,
            0.0,
            0.01,
        )
        torch.nn.init.normal_(
            self.brdf_lat_vecs.weight.data,
            0.0,
            0.01,
        )
        latent_size = 50
        self.calibration_lat_vecs = torch.nn.Embedding(max(55, self.num_id * 11), latent_size, max_norm=1.0)
        torch.nn.init.normal_(
            self.calibration_lat_vecs.weight.data,
            0.0,
            0.1,
        )
        self.loss_function = IDRLoss(device, config, **config.loss)

        self.log_num = -1
        # init the lower orders from NERF-OSR
        default_l = torch.tensor([[2.9861e+00, 3.4646e+00, 3.9559e+00],
            [1.0013e-01, -6.7589e-02, -3.1161e-01],
            [-8.2520e-01, -5.2738e-01, -9.7385e-02],
            [2.2311e-03, 4.3553e-03, 4.9501e-03],
            [-6.4355e-03, 9.7476e-03, -2.3863e-02],
            [1.1078e-01, -6.0607e-02, -1.9541e-01],
            [7.9123e-01, 7.6916e-01, 5.6288e-01],
            [6.5793e-02, 4.3270e-02, -1.7002e-01],
            [-7.2674e-02, 4.5177e-02, 2.2858e-01]]).float()
        default_l = default_l.reshape(1, -1)
        high_l = torch.zeros(1, 112 * 3).float()
        default_l = torch.cat([default_l, high_l], dim=-1)
        default_l = default_l.repeat(self.num_id, 1)
        self.sh_light = torch.nn.Embedding.from_pretrained(default_l, freeze=False)

    def forward(self, model_input):
        res = self.model(model_input)  # num_layers result
        return res

    def setup(self, stage):
        dataset = FaceScape(self.config, self.config.dataset.data_dir, self.config.dataset.factor, split='train',
                            facescape_id=self.config.dataset.facescape_id, facescape_exp=self.config.dataset.facescape_exp)
        val_dataset = FaceScape(self.config, self.config.dataset.data_dir, self.config.dataset.factor, split='test',
                                facescape_id=self.config.dataset.facescape_id, facescape_exp=self.config.dataset.facescape_exp)
        val_dataset.change_sampling_idx(-1)
        dataset.change_sampling_idx(self.config.train.num_pixels)
        self.train_dataset = dataset
        self.val_dataset = val_dataset

    def configure_optimizers(self):
        lr = float(self.config.train.lr)
        params = [{'name': 'render', 'params': self.rendering_network.parameters()},
                  {'name': 'calibration_lat_vecs', 'params': self.calibration_lat_vecs.parameters()},
                  {'name': 'imface', 'params': self.decoder.parameters(), 'lr': lr * self.config.get("template_loss_scale", 1.0)},
                  {'name': 'id_latent', 'params': self.id_lat_vecs.parameters()},
                  {'name': 'exp_latent', 'params': self.exp_lat_vecs.parameters()},
                  {'name': 'brdf_latent', 'params': self.brdf_lat_vecs.parameters()},
                  {'name': 'sh_light', 'params': self.sh_light.parameters()}]
        if not self.is_stage1:
            params.append({'name': 'residual', 'params': self.residual.parameters()})

        optimizer = torch.optim.Adam(params, lr=lr)
        sched_milestones = self.config.train.sched_milestones
        sched_factor = self.config.train.sched_factor
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, sched_milestones, gamma=sched_factor)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.dataset.batch_size,
                          shuffle=True,
                          collate_fn=self.train_dataset.collate_fn
                          )

    def val_dataloader(self):
        # must give 1 worker
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          collate_fn=self.val_dataset.collate_fn
                          )

    def on_train_epoch_start(self):
        self.train_dataset.change_sampling_idx(self.config.train.num_pixels)
        self.log_num = -1

    def training_step(self, batch, batch_nb):
        self.log_num = (self.log_num+1)
        indices, model_input, ground_truth = batch
        id_num = brdf_num = model_input["id"]
        exp_num = model_input["exp"]

        model_input['epoch'] = self.current_epoch
        id_latent = self.id_lat_vecs(id_num)
        exp_latent = self.exp_lat_vecs(exp_num)
        brdf_latent = self.brdf_lat_vecs(brdf_num)
        model_input['id_latent'] = id_latent
        model_input['exp_latent'] = exp_latent
        model_input['brdf_latent'] = brdf_latent
        calibration_code = self.calibration_lat_vecs(indices)
        model_input['calibration_code'] = calibration_code

        sh_light = self.sh_light(id_num.to(calibration_code.device)).reshape(-1, 121, 3)
        model_input['sh_light'] = sh_light

        output = self(model_input)

        loss_output = self.loss_function(output, ground_truth, id_latent, exp_latent, brdf_latent)
        loss = loss_output['loss']
        if self.log_num == 0:
            self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
            self.log('train/loss', loss)
            self.log('train/id_loss', loss_output['id_loss'], prog_bar=True)
            self.log('train/exp_loss', loss_output['exp_loss'], prog_bar=True)
            self.log('train/eikonal_loss', loss_output['eikonal_loss'], prog_bar=True)
            self.log('train/rgb_loss', loss_output['rgb_loss'], prog_bar=True)
            self.log('train/energy_loss', loss_output['energy_loss'], prog_bar=True)
            self.log('train/residual_loss', loss_output['residual_loss'], prog_bar=True)
            self.log('train/lgt_loss', loss_output['lgt_loss'], prog_bar=True),
        return loss

    def validation_step(self, batch, batch_nb):
        torch.set_grad_enabled(True)
        indices, model_input, ground_truth = batch
        h = self.val_dataset.h[indices]
        w = self.val_dataset.w[indices]
        img_res = [h, w]
        total_pixels = h * w
        id_num = brdf_num = model_input["id"]
        exp_num = model_input["exp"]

        id_latent = self.id_lat_vecs(id_num)
        exp_latent = self.exp_lat_vecs(exp_num)
        brdf_latent = self.brdf_lat_vecs(brdf_num)
        
        model_input['id_latent'] = id_latent
        model_input['exp_latent'] = exp_latent
        model_input['brdf_latent'] = brdf_latent
        calibration_code = self.calibration_lat_vecs(indices)
        model_input['calibration_code'] = calibration_code

        sh_light = self.sh_light(torch.tensor([id_num]).to(calibration_code.device)).reshape(-1, 121, 3)
        model_input['sh_light'] = sh_light

        split = utils.split_input(model_input, total_pixels)
        res = []

        for s in split:
            out = self(s)
            res.append({
                'points': out['points'].detach(),
                'rgb_values': out['rgb_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
            })
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        rgb_gt = ground_truth['rgb']
        # arrange data to plot
        batch_size, num_samples, _ = rgb_gt.shape

        object_mask = model_outputs['object_mask']
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)

        # plot rendered images
        rgb, gt = self.plot_image(rgb_eval, rgb_gt, img_res)

        rgb_eval = rgb_eval * object_mask.reshape(batch_size, num_samples, 1)
        rgb_gt = rgb_gt * object_mask.reshape(batch_size, num_samples, 1)
        val_psnr_fine = self.calc_psnr(rgb_gt, rgb_eval, object_mask)

        log = {'val/psnr': val_psnr_fine}
        img_gt = gt.squeeze(0).cpu()  # (3, H, W)
        coarse_rgb = rgb.squeeze(0).cpu()

        stack = torch.stack([img_gt, coarse_rgb])  # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_pred',
                                          stack, self.global_step)
        return log

    def plot_image(self, rgb_points, ground_true, img_res):
        batch_size, num_samples, channels = ground_true.shape
        gt = ground_true.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

        batch_size, num_samples, channels = rgb_points.shape
        rgb = rgb_points.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

        return rgb, gt

    def calc_psnr(self, x: torch.Tensor, y: torch.Tensor, mask):
        """
        Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
        """
        mse = torch.mean((x - y) ** 2) * x.shape[0] * x.shape[1] / mask.sum()
        psnr = -10.0 * torch.log10(mse)
        return psnr

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/psnr', mean_psnr, prog_bar=True)



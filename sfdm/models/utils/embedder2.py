import torch

# PREF code was taken from https://arxiv.org/pdf/2205.13524.pdf
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.reshape(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val
class PREF(nn.Module):
    def __init__(self, linear_freqs=[128]*3, reduced_freqs=[1]*3, feature_dim=16, sampling='linear', device='cuda') -> None:
        """
        Notice that a 3D phasor volume is viewed as  2D full specturm and 1D reduced specturm.
        Args: 
            linear_freqs: number of 2D freqeuncies 
            reduced_freqs: number of 1D frequencies 
            sampling: linear or explonential increasing
            feature_dim: output dimension
        """
        super().__init__()
        self.device = device
        self.linear_res = torch.tensor(linear_freqs).to(self.device)
        self.reduced_res = torch.tensor(reduced_freqs).to(self.device)
        
        if sampling == 'linear':
            self.axis = [torch.tensor([0.]+[i+1 for i in torch.arange(d-1)]).to(self.device) for d in reduced_freqs]
        else:
            self.axis = [torch.tensor([0.]+[2**i for i in torch.arange(d-1)]).to(self.device) for d in reduced_freqs]
        
        self.ktraj = self.compute_ktraj(self.axis, self.linear_res)

        self.output_dim = feature_dim
        self.out_dim = self.output_dim
        self.alpha_params = nn.Parameter(torch.tensor([1e-3]).to(self.device))
        self.params = nn.ParameterList(
            self.init_phasor_volume()
            )
        print(self)

    @property
    def alpha(self):
        # adaptively adjust the scale of phasors' magnitude during optimization.
        # not so important when Parsvel loss is imposed.
        return F.softplus(self.alpha_params, beta=10, threshold=1)

    @property
    def phasor(self):
        feature = [feat * self.alpha for feat in self.params]
        return feature

    def forward(self, inputs, bound=1):
        # map to [-1, 1]
        # inputs = inputs / bound
        xmax = inputs.max()
        xmin = inputs.min()
        b, a = 1.0, -1.0
        inputs = a + (b-a) / (xmax-xmin) * (inputs - xmin)
        # obtain embedding from phasor volume
        feature = self.compute_fft(self.phasor, inputs, interp=False)
        return feature.T

    # naive impl of inverse fourier transform
    def compute_spatial_volume(self, features):
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in self.res]
        Fx, Fy, Fz = features
        Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx = irfft(torch.fft.ifftn(Fx, dim=(3,4), norm='forward'), xx, ff=kx, T=Nx, dim=2)
        fy = irfft(torch.fft.ifftn(Fy, dim=(2,4), norm='forward'), yy, ff=ky, T=Ny, dim=3)
        fz = irfft(torch.fft.ifftn(Fz, dim=(2,3), norm='forward'), zz, ff=kz, T=Nz, dim=4)
        return (fx, fy, fz)

    # approx IFT as depicted in Eq.5 https://arxiv.org/pdf/2205.13524.pdf
    def compute_fft(self, features, xyz_sampled, interp=True):
        if interp:
            # using interpolation to compute fft = (N*N) log (N) d  + (N*N*d*d) + Nsamples 
            fx, fy, fz = self.compute_spatial_volume(features)
            volume = fx+fy+fz
            points = F.grid_sample(volume, xyz_sampled[None, None, None].flip(-1), align_corners=True).view(-1, *xyz_sampled.shape[:1],)
            # this is somewhat expensive when the xyz_samples is few and a 3D volume stills need computed
        else:
            # this is fast because we did 2d transform and matrix multiplication . (N*N) logN d + Nsamples * d*d + 3 * Nsamples 
            Nx, Ny, Nz = self.linear_res
            Fx, Fy, Fz = features
            d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
            kx, ky, kz = self.axis
            kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
            xs, ys, zs = xyz_sampled.chunk(3, dim=-1)
            Fx = torch.fft.ifftn(Fx, dim=(3,4), norm='forward')
            Fy = torch.fft.ifftn(Fy, dim=(2,4), norm='forward')
            Fz = torch.fft.ifftn(Fz, dim=(2,3), norm='forward')
            fx = grid_sample_cmplx(Fx.transpose(3,3).flatten(1,2), torch.stack([zs, ys], dim=-1)[None]).reshape(Fx.shape[1], Fx.shape[2], -1)
            fy = grid_sample_cmplx(Fy.transpose(2,3).flatten(1,2), torch.stack([zs, xs], dim=-1)[None]).reshape(Fy.shape[1], Fy.shape[3], -1)
            fz = grid_sample_cmplx(Fz.transpose(2,4).flatten(1,2), torch.stack([xs, ys], dim=-1)[None]).reshape(Fz.shape[1], Fz.shape[4], -1)
            fxx = batch_irfft(fx, xs, kx, Nx)
            fyy = batch_irfft(fy, ys, ky, Ny)
            fzz = batch_irfft(fz, zs, kz, Nz)
            return fxx+fyy+fzz

        return points

    @torch.no_grad()
    def init_phasor_volume(self):
        # rough approximation 
        # transform the fourier domain to spatial domain
        Nx, Ny, Nz = self.linear_res
        d1, d2, d3 = self.reduced_res
        # xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (d1,d2,d3)]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (Nx,Ny,Nz)]
        XX, YY, ZZ = [torch.linspace(0, 1, N).to(self.device) for N in (Nx,Ny,Nz)]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        
        fx = torch.ones(1, self.output_dim, len(xx), Ny, Nz).to(self.device)
        fy = torch.ones(1, self.output_dim, Nx, len(yy), Nz).to(self.device)
        fz = torch.ones(1, self.output_dim, Nx, Ny, len(zz)).to(self.device)
        normx = torch.stack(torch.meshgrid([2*xx-1, 2*YY-1, 2*ZZ-1]), dim=-1).norm(dim=-1)
        normy = torch.stack(torch.meshgrid([2*XX-1, 2*yy-1, 2*ZZ-1]), dim=-1).norm(dim=-1)
        normz = torch.stack(torch.meshgrid([2*XX-1, 2*YY-1, 2*zz-1]), dim=-1).norm(dim=-1)

        fx = fx * normx[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fy = fy * normy[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fz = fz * normz[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))

        fxx = rfft(torch.fft.fftn(fx.transpose(2,4), dim=(2,3), norm='forward'),xx, ff=kx, T=Nx).transpose(2,4)
        fyy = rfft(torch.fft.fftn(fy.transpose(3,4), dim=(2,3), norm='forward'),yy, ff=ky, T=Ny).transpose(3,4)
        fzz = rfft(torch.fft.fftn(fz.transpose(4,4), dim=(2,3), norm='forward'),zz, ff=kz, T=Nz).transpose(4,4)
        return [torch.nn.Parameter(fxx), torch.nn.Parameter(fyy), torch.nn.Parameter(fzz)]


    def compute_ktraj(self, axis, res): # the associated frequency coordinates.
        ktraj2d = [torch.fft.fftfreq(i, 1/i).to(self.device) for i in res]
        ktraj1d = [torch.arange(ax).to(torch.float).to(self.device) if type(ax) == int else ax for ax in axis]
        ktrajx = torch.stack(torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1)
        ktrajy = torch.stack(torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1)
        ktrajz = torch.stack(torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1)
        ktraj = [ktrajx, ktrajy, ktrajz]
        return ktraj

    def parseval_loss(self):
        # Parseval Loss
        new_feats = [Fk.reshape(-1, *Fk.shape[2:],1) * 1j * np.pi * wk.reshape(1, *Fk.shape[2:], -1) 
            for Fk, wk in zip(self.phasor, self.ktraj)]
        loss = sum([feat.abs().square().mean() for feat in itertools.chain(*new_feats)])
        return loss


## utilis 
def irfft(phasors, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    phasors = phasors.transpose(dim, -1)
    assert phasors.shape[-1] == len(ff) if ff is not None else True
    device = phasors.device
    xx = xx * (T-1) / T                       # to match torch.fft.fft
    N = phasors.shape[-1]
    if ff is None:
        ff = torch.arange(N).to(device)       # positive freq only
    xx = xx.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * xx * ff).to(device)
    M = M * ((ff>0)+1)[None]                  # Hermittion symmetry
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


def batch_irfft(phasors, xx, ff, T):
    # numerial integration 
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx+1) * 0.5
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j*np.pi*xx * ff)                   # twiddle factor
    twiddle = twiddle * ((ff > 0)+1)[None]                  # hermitian # [N, d]
    twiddle = twiddle.transpose(0,1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)

def rfft(spatial, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1/T) # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out

def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    # sampled = F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
    #         1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)
    # print('input=',input.shape,'grid=', grid.shape)
    sampled = grid_sample(input.real, grid) + \
            1j * grid_sample(input.imag, grid)
    # exit(0)
    return sampled

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'alpha' in self.kwargs:
            self.alpha = self.kwargs['alpha']
        else:
            self.alpha = None
        self.create_embedding_fn()

    def get_scalar(self, j, L):
        """j \in [0,L-1], L is frequency length, was taken form paper(improved surface reconstruction...)"""
        # return (1.0-torch.cos(torch.clamp(self.alpha*L-j,0,1)*math.pi)) / 2.0
        return 1.0

    def update_alpha(self, alpha=1.):
        if self.alpha is None:
            self.alpha = alpha
        else:
            # self.alpha = torch.clamp(self.alpha+alpha, min=0., max=1.)
            self.alpha = max(self.alpha + alpha, 0.)
            self.alpha = min(self.alpha, 1.)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, j=torch.log(freq), L=N_freqs, p_fn=p_fn,
                                 freq=freq: self.get_scalar(j, L)*p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, alpha=1, input_dims=3):
    embed_kwargs = {
        'alpha': alpha,
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim
    # def embed(x, eo=embedder_obj): return eo.embed(x)
    # return embed, embedder_obj.out_dim

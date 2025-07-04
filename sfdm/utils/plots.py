import mcubes
import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from sfdm.utils import rend_util

def plot(device, model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, id_latent, exp_latent, plot_nimgs, max_depth, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    # points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)

    # depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    # depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    # depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size,-1)

    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)

    # plot rendered images
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    # plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

    data = []

    # plot surface
    # surface_traces = get_surface_trace(path=path,
    #                                    epoch=epoch,
    #                                    sdf=lambda x, y, z: model(x.unsqueeze(0), z, y),
    #                                    id_latent=id_latent,
    #                                    exp_latent=exp_latent,
    #                                    resolution=resolution
    #                                    )

def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, id_latent, exp_latent, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts, id_latent, exp_latent).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(id_latent, exp_latent, density_fun, sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts, exp_latent, id_latent).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))
    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces)
    mesh_low_res.export('sdf.ply')
    condition1 = np.abs(z) > 0.1
    condition2 = np.abs(z) <= 0.1
    z[condition1] = 0
    z[condition2] = 1
    z = z.astype(bool)
    density = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        dir = torch.zeros_like(pnts).cuda()
        density.append(density_fun(pnts, dir, id_latent, exp_latent).detach().cpu().numpy())
    density = np.concatenate(density, axis=0)
    density = density.astype(np.float32)
    density = np.maximum(density, 0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=density.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=15,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]),
        mask=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]))
    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
    mesh_low_res = trimesh.Trimesh(verts, faces)
    mesh_low_res.export('density.ply')
    vector_len = np.linalg.norm(points.detach().cpu().numpy(), axis=1)
    mask = np.where(vector_len <= 1, 1, 0).astype(np.int).reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                           grid['xyz'][2].shape[0])
    outd = density.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                           grid['xyz'][2].shape[0])
    outz = z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0], grid['xyz'][2].shape[0])
    print(outd[outd > 0].shape)
    outd = outd * outz * mask
    print(outd[outd > 0].shape)
    np.save('density.npy', outd)


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = (ground_true.cuda() + 1.) / 2.
    rgb_points = (rgb_points + 1. ) / 2.

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

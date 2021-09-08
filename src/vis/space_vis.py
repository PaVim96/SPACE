from torch.utils.tensorboard import SummaryWriter
import imageio
import numpy as np
import torch

import matplotlib
from utils import spatial_transform
from .utils import bbox_in_one, colored_bbox_in_one_image
from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
from collections import Counter
from torchvision.utils import draw_bounding_boxes as draw_bb
from PIL import Image
import PIL
from eval import convert_to_boxes, read_boxes

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SpaceVis:
    @torch.no_grad()
    def train_vis(self, model, writer: SummaryWriter, log, global_step, mode, cfg, dataset, num_batch=10):
        """
        """
        writer.add_scalar(f'{mode}/z_what_delta', torch.sum(log['z_what_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/z_what_loss_pool', torch.sum(log['z_what_loss_pool']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/z_what_loss_objects', torch.sum(log['z_what_loss_objects']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/z_pres_loss', torch.sum(log['z_pres_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss', torch.sum(log['flow_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/objects_detected', torch.sum(log['objects_detected']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/total_loss', log['loss'], global_step=global_step)
        # FYI: For visualization only use some images of each stack in the batch
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][:num_batch]
        log_img = AttrDict(log)

        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log_img.fg, log_img.z_pres, log_img.z_scale, log_img.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log_img.imgs[:, None]
        fg = log_img.fg[:, None]
        recon = log_img.y[:, None]
        fg_box = fg_box[:, None]
        bg = log_img.bg[:, None]
        # (B, K, 3, H, W)
        comps = log_img.comps
        # (B, K, 3, H, W)
        masks = log_img.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log_img.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.reshape(B * N, 3, H, W)

        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/0-separations', grid_image, global_step)

        grid_image = make_grid(log_img.imgs, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/1-image', grid_image, global_step)

        grid_image = make_grid(log_img.y, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/2-reconstruction_overall', grid_image, global_step)

        grid_image = make_grid(log_img.bg, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/3-background', grid_image, global_step)

        grid_flow = log_img.grid_flow.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        flow_image = log_img.imgs * grid_flow
        grid_image = make_grid(flow_image, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/4-flow', grid_image, global_step)

        count = log_img.z_pres.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        loss = log['loss'].mean()
        writer.add_scalar(f'{mode}/loss', loss, global_step=global_step)
        writer.add_scalar(f'{mode}/count', count, global_step=global_step)

        mse = (log_img.y - log_img.imgs) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        writer.add_scalar(f'{mode}/mse', mse, global_step=global_step)
        writer.add_scalar(f'{mode}/log_like', log_img['log_like'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/What_KL', log_img['kl_z_what'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Where_KL', log_img['kl_z_where'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Pres_KL', log_img['kl_z_pres'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Depth_KL', log_img['kl_z_depth'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Bg_KL', log_img['kl_bg'].mean(), global_step=global_step)

        z_pres_prob = log_img.z_pres_prob.reshape(log_img.grid_flow.shape)
        z_pres_prob = z_pres_prob.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        z_pres_image = log_img.imgs * z_pres_prob
        grid_image = make_grid(z_pres_image, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/5-z_pres', grid_image, global_step)

        bb_image = draw_image_bb(model, cfg, dataset, global_step)
        grid_image = make_grid(bb_image, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/6-bounding_boxes', grid_image, global_step)

    @torch.no_grad()
    def show_vis(self, model, dataset, indices, path, device):
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        data = next(iter(dataloader))
        data = data.to(device)
        loss, log = model(data, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B * N, 3, H, W)

        # (3, H, W)
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)

        # (H, W, 3)
        image = torch.clamp(grid_image, 0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(path, image)

    @torch.no_grad()
    def show_bb(self, model, image, path, device):
        image = image.to(device)
        loss, log = model(image, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = colored_bbox_in_one_image(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        plt.imshow(fg_box[0][0].permute(1, 2, 0))
        plt.show()

def draw_image_bb(model, cfg, dataset, global_step):
    indices = np.random.choice(len(dataset), size=10, replace=False)
    png_indices = [4 * i + dataset.flow for i in indices]
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    data = next(iter(dataloader))
    data = data.to(cfg.device)
    loss, log = model(data, global_step)
    bb_path = f"../aiml_atari_data/space_like/{cfg.gamelist[0]}/train/bb"
    rgb_folder = f"../aiml_atari_data/rgb/{cfg.gamelist[0]}/train"
    boxes_gt = read_boxes(bb_path, 128, indices=indices)
    boxes_pred = []
    z_where, z_pres_prob = log['z_where'][:10], log['z_pres_prob'][:10]
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.5
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
    boxes_pred.extend(boxes_batch)
    result = []
    for idx, gt, pred in zip(png_indices, boxes_gt, boxes_pred):
        pil_img = Image.open(f'{rgb_folder}/{idx:05}.png', ).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        image = np.array(pil_img)
        objects = torch.from_numpy(np.loadtxt(f'{bb_path}/bb_{idx}.txt', delimiter=',', usecols=[0, 1, 2, 3]))
        objects[:, 2:] += objects[:, :2]
        torch_img = torch.from_numpy(image).permute(2, 0, 1)
        bb_img = draw_bb(torch_img, objects, colors=['red']*len(objects)).permute(0, 2, 1)
        pred_tensor = torch.FloatTensor(pred) * 128
        pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        bb_img = draw_bb(bb_img, pred_tensor, colors=['green']*len(pred_tensor)).permute(0, 2, 1)
        result.append(bb_img)
    return torch.stack(result)

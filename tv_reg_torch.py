import torch
import numpy as np
from matplotlib import pyplot as plt

import config
from model import PhaseObject3D


configs = config.Configs()

def tv_reg_3d(object, iter=10, t=0.1, ep=1, lamda=0.5):
    tv_object = PhaseObject3D(configs.phase_obj_shape, configs.pad_size, configs.n_media, configs.n_min, configs.n_max)
    tv_object.RI_pad_cuda = object.RI_pad_cuda
    
    im = object.RI_cuda
    imtmp = im
    for i in range(iter):
        ux  = torch.roll(imtmp, 1, dims=1)[1:-1, 1:-1, :] - torch.roll(imtmp, -1, dims=1)[1:-1, 1:-1, :]
        uy  = torch.roll(imtmp, 1, dims=0)[1:-1:,1:-1, :] - torch.roll(imtmp, -1, dims=0)[1:-1:,1:-1, :]
        uxx = torch.roll(imtmp, 1, dims=1)[1:-1, 1:-1, :] + torch.roll(imtmp, -1, dims=1)[1:-1, 1:-1, :] - 2*imtmp[1:-1, 1:-1, :]
        uyy = torch.roll(imtmp, 1, dims=0)[1:-1, 1:-1, :] + torch.roll(imtmp, -1, dims=0)[1:-1, 1:-1, :] - 2*imtmp[1:-1, 1:-1, :]
        uxy = torch.roll(torch.roll(imtmp, 1, dims=0), 1, dims=1)[1:-1, 1:-1, :] + torch.roll(torch.roll(imtmp, -1, dims=0),-1, dims=1)[1:-1, 1:-1, :] 
        -     torch.roll(torch.roll(imtmp, 1, dims=0),-1, dims=1)[1:-1, 1:-1, :] - torch.roll(torch.roll(imtmp, -1, dims=0), 1, dims=1)[1:-1, 1:-1, :]
        divp = ((uy * uy + ep) * uxx - 2* ux * uy * uxy + (ux * ux + ep) * uyy) / ((ux * ux + uy * uy + ep))
        imtmp[1:-1, 1:-1, :] = im[1:-1, 1:-1, :]

        ux  = torch.roll(imtmp, 1, dims=1)[:, 1:-1, 1:-1] - torch.roll(imtmp, -1, dims=1)[:, 1:-1, 1:-1]
        uz  = torch.roll(imtmp, 1, dims=2)[:, 1:-1, 1:-1] - torch.roll(imtmp, -1, dims=2)[:, 1:-1, 1:-1]
        uxx = torch.roll(imtmp, 1, dims=1)[:, 1:-1, 1:-1] + torch.roll(imtmp, -1, dims=1)[:, 1:-1, 1:-1] - 2*imtmp[:, 1:-1, 1:-1]
        uzz = torch.roll(imtmp, 1, dims=2)[:, 1:-1, 1:-1] + torch.roll(imtmp, -1, dims=2)[:, 1:-1, 1:-1] - 2*imtmp[:, 1:-1, 1:-1]
        uxz = torch.roll(torch.roll(imtmp, 1, dims=2), 1, dims=1)[:, 1:-1, 1:-1] + torch.roll(torch.roll(imtmp, -1, dims=2),-1, dims=1)[:, 1:-1, 1:-1] 
        -     torch.roll(torch.roll(imtmp, 1, dims=2),-1, dims=1)[:, 1:-1, 1:-1] - torch.roll(torch.roll(imtmp, -1, dims=2), 1, dims=1)[:, 1:-1, 1:-1]
        divp = ((uz * uz + ep) * uxx - 2* ux * uz * uxz + (ux * ux + ep) * uzz) / ((ux * ux + uz * uz + ep))
        imtmp[:, 1:-1, 1:-1] += t *( lamda * (im[:, 1:-1, 1:-1] - imtmp[:, 1:-1, 1:-1]) + divp)

        uz  = torch.roll(imtmp, 1, dims=2)[1:-1, :, 1:-1] - torch.roll(imtmp, -1, dims=2)[1:-1, :, 1:-1]
        uy  = torch.roll(imtmp, 1, dims=0)[1:-1, :, 1:-1] - torch.roll(imtmp, -1, dims=0)[1:-1, :, 1:-1]
        uzz = torch.roll(imtmp, 1, dims=2)[1:-1, :, 1:-1] + torch.roll(imtmp, -1, dims=2)[1:-1, :, 1:-1] - 2*imtmp[1:-1, :, 1:-1]
        uyy = torch.roll(imtmp, 1, dims=0)[1:-1, :, 1:-1] + torch.roll(imtmp, -1, dims=0)[1:-1, :, 1:-1] - 2*imtmp[1:-1, :, 1:-1]
        uzy = torch.roll(torch.roll(imtmp, 1, dims=0), 1, dims=2)[1:-1, :, 1:-1] + torch.roll(torch.roll(imtmp, -1, dims=0),-1, dims=2)[1:-1, :, 1:-1] 
        -     torch.roll(torch.roll(imtmp, 1, dims=0),-1, dims=2)[1:-1, :, 1:-1] - torch.roll(torch.roll(imtmp, -1, dims=0), 1, dims=2)[1:-1, :, 1:-1]
        divp = ((uy * uy + ep) * uzz - 2* uz * uy * uzy + (uz * uz + ep) * uyy) / ((uz * uz + uy * uy + ep))
        imtmp[1:-1, :, 1:-1] += t *( lamda * (im[1:-1, :, 1:-1] - imtmp[1:-1, :, 1:-1]) + divp)

    tv_object.RI_pad_cuda[configs.pad_size+1:-configs.pad_size-1, configs.pad_size+1:-configs.pad_size-1, :] = imtmp[1:-1, 1:-1, :]
    return tv_object

    
def tv3d_loss(model):
    param = torch.Tensor(config.slice_size, config.slice_size, config.slice_num[-1])
    i = 0
    for slice_name, slice in model.SliceLayer_dic.items():
        param[i] = slice.weights.data
        i += 1
    param_dx = torch.abs(torch.roll(param, shifts=1, dims=0) - param)
    param_dy = torch.abs(torch.roll(param, shifts=1, dims=1) - param)
    param_dz = torch.abs(torch.roll(param, shifts=1, dims=2) - param)
    param_dx = torch.sum(param_dx, dim=0)
    param_dy = torch.sum(param_dy, dim=1)
    param_dz = torch.sum(param_dz, dim=2)
    param_dx_norm = torch.norm(param_dx, p=2)
    param_dy_norm = torch.norm(param_dy, p=2)
    param_dz_norm = torch.norm(param_dz, p=2)
    tv_regularization_loss = param_dx_norm + param_dy_norm + param_dz_norm
    return tv_regularization_loss
    


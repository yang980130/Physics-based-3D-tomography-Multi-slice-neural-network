from lib2to3.pgen2.grammar import Grammar
from logging import raiseExceptions
import os
from cv2 import DFT_COMPLEX_INPUT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.tensorboard.writer import SummaryWriter

from math import pi
from config import Configs
from model import PhaseObject3D
from tv_reg_torch import tv_reg_3d

import time

config = Configs()

class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()


class IncidentWavefrontLayer(MultiplyLayer):
    def __init__(self):
        super(IncidentWavefrontLayer, self).__init__()
        N = config.slice_pad_size
        self.x = config.HR_pixel_size*torch.linspace(-N/2, N/2-1, N)
        self.yy, self.xx = torch.meshgrid(self.x, self.x)
        self.xx = self.xx.to(config.device)
        self.yy = self.yy.to(config.device)
    def forward(self, fxfy):
        output = torch.exp(1j*2*pi*(fxfy[0]*self.xx + fxfy[1]*self.yy))
        return output


class SliceLayer(MultiplyLayer):
    def __init__(self):
        super(SliceLayer, self).__init__()
        M = config.slice_pad_size
        self.weights = Parameter(torch.Tensor(M, M), requires_grad=config.en_optim_object)
        self.coeff = Parameter(torch.Tensor(1), requires_grad=config.en_optim_object)
        return

    def forward(self, input, z, wlength):
        slice = self.weights
        if config.scatter_mode == 'MS' or config.scatter_mode =='MLB':
            output = input * torch.exp(1j*2*pi*(slice)*z/wlength)
        elif config.scatter_mode == 'MS_LIAC':
            output = self.coeff[0] * input * torch.exp(1j*2*pi*(slice)*z/wlength)
        else:
            print('The scatter mode setting is wrong!!')
            raise
        weights = self.weights[config.pad_size: -config.pad_size, config.pad_size: -config.pad_size]
        return output, self.weights

class BackScattering(MultiplyLayer):
    def __init__(self):
        super(BackScattering, self).__init__()
        self.M = config.slice_pad_size
        psize = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        self.coeff = Parameter(torch.ones(config.slice_num_list[-1]), requires_grad=True).to(config.device)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        dis = self.sqrt__((config.n_media/config.wlength)**2-torch.sqrt((fxx**2+fyy**2))).to(config.device)

        self.prop_phs = 1j*2*pi*self.sqrt__((config.n_media/config.wlength)**2-(fxx**2+fyy**2)).to(config.device)
        self.Fgreen = (-1j * torch.exp(self.prop_phs*config.imaging_depth/config.slice_num_list[-1]) / (4*pi*dis)).to(config.device)

    def forward(self, Uin, n_):
        V = (2*pi/config.wlength)**2*(-n_)*(n_+2*config.n_media)
        return torch.fft.ifft2(self.Fgreen * torch.fft.fft2(Uin*V))

    def sqrt__(self, x):
        mask_1   = (x >= 0)
        mask_2   = (x <  0)
        x[mask_2] = -x[mask_2]
        positive_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        negative_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        positive_sqrt = torch.sqrt(x) * mask_1
        negative_sqrt = 1j * torch.sqrt(x) * mask_2
        output = positive_sqrt + negative_sqrt
        return output


class AngularSpecteumDiffractionLayer(MultiplyLayer):
    def __init__(self):
        super(AngularSpecteumDiffractionLayer, self).__init__()
        self.M    = config.slice_pad_size
        psize     = config.HR_pixel_size
        t_k0      = config.k0 * config.n_media / (2*pi)
        kx          = torch.linspace(-pi/psize, pi/psize, self.M+1)[0:self.M] / (2*pi)
        [kxx, kyy]  = torch.meshgrid(kx, kx)
        kz_map_square = t_k0**2-kxx**2-kyy**2
        self.kz_map = self.sqrt__(kz_map_square).to(config.device)
        return
    
    def sqrt__(self, x):
        mask_1   = (x >= 0)
        mask_2   = (x <  0)
        x[mask_2] = -x[mask_2]
        positive_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        negative_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        positive_sqrt = torch.sqrt(x) * mask_1
        negative_sqrt = 1j * torch.sqrt(x) * mask_2
        output = positive_sqrt + negative_sqrt
        return output

    def forward(self, input, z):
        kzm    = self.kz_map
        output = input * torch.exp(1j*kzm*z)
        return output

    def cuda_init(self):
        self.kz_map = self.kz_map.to(config.device)
        self.mask   = self.mask  .to(config.device)


class MultiSliceAngularSpecteumDiffractionLayer(MultiplyLayer):
    def __init__(self, wlength):
        super(MultiSliceAngularSpecteumDiffractionLayer, self).__init__()
        self.M     = config.slice_pad_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fyy, fxx] = (torch.meshgrid(fx, fx))
        fx         = torch.fft.ifftshift(fx)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        self.prop_phs   = 1j*2*pi*self.sqrt__((config.n_media/wlength)**2-(fxx**2+fyy**2)).to(config.device)
        self.prop_phs_x = (1j*2*pi*fxx/config.led_height).to(config.device)
        self.prop_phs_y = (1j*2*pi*fyy/config.led_height).to(config.device)
        return
    
    def sqrt__(self, x):
        mask_1   = (x >= 0)
        mask_2   = (x <  0)
        x[mask_2] = -x[mask_2]
        positive_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        negative_sqrt = torch.zeros([self.M, self.M], dtype=torch.complex64)
        positive_sqrt = torch.sqrt(x) * mask_1
        negative_sqrt = 1j * torch.sqrt(x) * mask_2
        output = positive_sqrt + negative_sqrt
        return output

    def forward(self, input, z):
        if z >= 0:
            output = input * (torch.exp(self.prop_phs*z))
        else:
            output = input * torch.conj(torch.exp(self.prop_phs*(-z)))
        return output


class PupilLayer(MultiplyLayer):
    def __init__(self):
        super(PupilLayer, self).__init__()
        M         = config.slice_size
        m         = config.CTFsize
        self.pad  = (M - m) // 2
        if self.pad * 2 == M - m:
            pad_ = self.pad
        else:
            pad_ = self.pad + 1
        pad      = self.pad
        xy                     = torch.linspace(-1, 1, m)
        self.x_map, self.y_map = torch.meshgrid(xy, xy)
        self.x_map    = self.x_map.to(config.device)
        self.y_map    = self.y_map.to(config.device)
        self.window   = torch.ones([m, m])
        circular_mask = torch.square(self.x_map) + torch.square(self.y_map) >= 1
        self.window[circular_mask] = 0
        self.window   = F.pad(self.window, (pad,pad_,pad,pad_)).to(config.device)

    def forward(self, input):
        output = input * self.window
        return output
    

class MultiSlicePupilLayer(MultiplyLayer):
    def __init__(self, wlength):
        super(MultiSlicePupilLayer, self).__init__()
        self.M     = config.slice_pad_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.M*psize)
        fx         = dfx*torch.linspace(-self.M/2, self.M/2, self.M+1)[0:self.M]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        fx         = torch.fft.ifftshift(fx)
        fxx        = torch.fft.ifftshift(fxx)
        fyy        = torch.fft.ifftshift(fyy)
        NA_crop    = (fxx**2 + fyy**2) < (config.NA/wlength)**2
        self.NA_crop    = NA_crop.long().to(config.device)
        return 

    def forward(self, input):
        output = input*self.NA_crop
        return output


class MutiSliceCropPupilLayer(MultiplyLayer):
    def __init__(self, wlength):
        super(MutiSliceCropPupilLayer, self).__init__()
        self.M     = config.slice_pad_size
        self.m     = config.capture_size
        psize      = config.HR_pixel_size
        dfx        = 1/(self.m*psize)
        fx         = dfx*torch.linspace(-self.m/2, self.m/2, self.m+1)[0:self.m]
        [fxx, fyy] = (torch.meshgrid(fx, fx))
        NA_crop    = (fxx**2 + fyy**2) < (config.NA/wlength)**2
        self.NA_crop    = (NA_crop.long() + torch.zeros([self.m, self.m], dtype=torch.complex64)).to(config.device)
        return 

    def forward(self, input):
        output = input[(config.slice_pad_size-config.capture_size)//2:(config.slice_pad_size+config.capture_size)//2, (config.slice_pad_size-config.capture_size)//2:(config.slice_pad_size+config.capture_size)//2]*self.NA_crop
        return output


class LightCorrectionLayer(MultiplyLayer):
    def __init__(self):
        super(LightCorrectionLayer, self).__init__()
        N = config.num_illu
        self.weights = Parameter(torch.ones(N), requires_grad=config.en_optim_light)

    def forward(self, E_in, idx):
        E_out = self.weights[idx] * E_in
        return E_out


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        h_x = x.size()[0]
        w_x = x.size()[1]
        z_x = x.size()[2]
        count_h = self._tensor_size(x[1:,:,:])
        count_w = self._tensor_size(x[:,1:,:])
        count_z = self._tensor_size(x[:,:,1:])
        h_tv = torch.pow((x[1:,:,:]-x[:h_x-1,:,:]),2).sum()
        w_tv = torch.pow((x[:,1:,:]-x[:,:w_x-1,:]),2).sum()
        z_tv = torch.pow((x[:,:,1:]-x[:,:,:z_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+z_tv/count_z)

    def _tensor_size(self, t):
        return t.size()[0]*t.size()[1]*t.size()[2]


class MultiSliceNN(MultiplyLayer):
    def __init__(self, slice_num):
        super(MultiSliceNN, self).__init__()
        config  = Configs()
        self.slice_number = slice_num
        self.Dropout = nn.Dropout(0.1)
        self.Inputwave = IncidentWavefrontLayer()
        self.SliceLayer_dic = {}
        for i in range(slice_num):
            self.SliceLayer_dic['Slice_{}'.format(i)] = SliceLayer()
        self.Prop    = MultiSliceAngularSpecteumDiffractionLayer(config.wlength)
        self.BS       = BackScattering()
        self.Pupil   = MultiSlicePupilLayer(config.wlength)
        self.LightCor = LightCorrectionLayer()
        self.weights  = torch.zeros(config.slice_pad_size, config.slice_pad_size, config.slice_num_list[-1])
        return

    
    def forward(self, input, loc, i):
        Fourier_E1 = torch.fft.fft2(self.Inputwave(input))
        idx = 0
        for slice_name, slice in self.SliceLayer_dic.items():
            E1_ = torch.fft.ifft2(self.Prop(Fourier_E1, config.imaging_depth/self.slice_number))
            if config.scatter_mode == 'MLB':
                _, weights = slice(E1_, config.imaging_depth/self.slice_number, config.wlength)
                backward_scatter_field = self.BS(torch.fft.ifft2(Fourier_E1), weights)
                E1 = E1_ + backward_scatter_field
            else:
                E1, weights = slice(E1_, config.imaging_depth/self.slice_number, config.wlength)
            self.weights[:,:,idx] = weights
            
            idx += 1
            Fourier_E1 = torch.fft.fft2(E1)
        E1 = self.Prop(Fourier_E1, -config.imaging_depth/2)
        E1 = self.Pupil(E1)
        E1 = torch.fft.ifft2(E1)
        intensity = self.LightCor(torch.abs(E1), i)
        if config.pad_size == 0:
            return intensity, self.weights
        else:
            return intensity[config.pad_size:-config.pad_size, config.pad_size:-config.pad_size], self.weights


    def initModel(self, phaseobj_3d, mode='init'):
        i = 0
        coeff = len(self.SliceLayer_dic) // phaseobj_3d.RI_pad.shape[2]
        for slice_name, slice in self.SliceLayer_dic.items():
            if mode == 'init':
                idx = i // coeff
            elif mode == 'gt':
                idx = i
            else:
                raise('The param -- init mode is wrong!')
            slice.weights.data = torch.tensor(phaseobj_3d.RI_pad_cuda[..., idx]).type(config.torch_float).to(config.device)
            slice.coeff.data = torch.ones(1)
            i += 1
            self.BS.coeff.data = torch.ones(config.slice_num_list[-1]).to(config.device)

    def extractParameters2cpu(self):
        object_3d = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
        slice_dic = {}
        for slice_name, slice in self.SliceLayer_dic.items():
            slice_dic[slice_name] = slice.weights.data.to('cpu').detach().numpy()
        object_3d.createPhaseObject3DfromArrayDic(slice_dic)
        return object_3d

    def extractParameters2cuda(self):
        object_3d = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
        slice_dic = {}
        for slice_name, slice in self.SliceLayer_dic.items():
                slice_dic[slice_name] = slice.weights.data
        object_3d.createPhaseObject3DfromTensorDic(slice_dic)
        return object_3d


def generate_optimizer(model, lr):
    optimizer = torch.optim.Adam([
        {'params': x.parameters(), 'lr': lr} for x in model.SliceLayer_dic.values()])
    optimizer_ = torch.optim.Adam([
        {'params': model.LightCor.parameters(), 'lr': 1e-2},
        {'params': model.BS.parameters(), 'lr': 1e-2}])
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    schedular_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=10, gamma=0.9)
    optimizer_list = [optimizer, optimizer_]
    schedular_list = [schedular, schedular_]
    return optimizer_list, schedular_list


class Optimization():
    def __init__(self, model, optimizer, scheduler):
        config     = Configs()
        self.loss_fn    = nn.L1Loss() # nn.MSELoss()
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_list  = []
        self.num_illu  = config.num_illu

    def test(self, ledarray):
        input     = torch.tensor(ledarray.fxfy).to(config.device)
        loc       = torch.tensor(ledarray.loc).to(config.device)
        predict   = []
        for i in range(self.num_illu):
            output, _ = self.model(input[i], loc[i], i)
            output = output.detach()
            predict.append(output)
        return predict
    
    def train(self, ledarray, in_measurement):
        device    = config.device
        num_epoch = config.num_epoch
        optimizer = self.optimizer
        scheduler = self.scheduler
        target    = in_measurement
        input     = torch.tensor(ledarray.fxfy).to(device)
        loc       = torch.tensor(ledarray.loc).to(device)
        tvloss_fn = TVLoss(100) # 传入参数为tvloss的放大系数
        optim_start = time.time()
        ### 启用tensorboard监视训练 ###
        writer = SummaryWriter(comment='NN', filename_suffix='NN')
        for epoch in range(num_epoch):
            running_loss = 0
            epoch_start = time.time()
            for i in range(self.num_illu):
                output, _ = self.model(input[i], loc[i], i)
                loss   = self.loss_fn(output, target[i])
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                loss.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[1].step()
                running_loss = running_loss + loss.item()
            if config.tv:
            # if config.tv and epoch == config.num_epoch-1:
                object = self.model.extractParameters2cuda()
                tv_loss = tvloss_fn(object.RI_cuda)
                print('epoch:{}, before_tv:{}'.format(epoch+1, tv_loss.item()))
                tv_start = time.time()
                tv_object = tv_reg_3d(object, iter=config.tv_beta[0], t=config.tv_beta[1], ep=1, lamda=config.tv_beta[2])
                self.model.initModel(tv_object)
                _, weights = self.model([0, 0], [0], 0)
                tv_loss = tvloss_fn(weights)
                tv_end   = time.time()
                print('tv cost {}s'.format(tv_end-tv_start))
                print('after_tv:{}'.format(tv_loss.item()))
            else:
                print('epoch:{}'.format(epoch+1))

            scheduler[0].step()
            scheduler[1].step()
            epoch_loss = running_loss / self.num_illu
            self.loss_list.append(epoch_loss) # 这个loss_list可以 返回/输出
            ###tensorboard###
            if config.scatter_mode == 'MS':
                writer.add_scalars('loss', {'{}'.format('MS'): epoch_loss}, epoch)
            if config.scatter_mode == 'MS_LIAC':
                writer.add_scalars('loss', {'{}'.format('MS_LIAC'): epoch_loss}, epoch)
            if config.scatter_mode == 'MLB':
                writer.add_scalars('loss', {'{}'.format('MLB'): epoch_loss}, epoch)

            ###记录每个epoch的优化结果###
            # if (epoch+1) % 5 == 0 or epoch == 0:
            sample = self.model.extractParameters2cpu()
            sample.saveOptimObject(epoch+1)
            ###--------------------###
            print('loss:{}'.format(epoch_loss))    
            epoch_end = time.time()
            print('last epoch cost {}s'.format(epoch_end-epoch_start))
        optim_end = time.time()
        print('total optim cost {}s'.format(optim_end-optim_start))
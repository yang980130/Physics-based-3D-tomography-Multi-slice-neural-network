import numpy as np
import torch
import torch.nn.functional as F
from math import pi
import cv2
from PIL import Image

import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.io as io
import h5py

import config
from config import Configs

np_complex_datatype = config.np_complex_datatype
np_float_datatype   = config.np_float_datatype

config = Configs()

class Slice():
    def __init__(self, size):
        super(Slice, self).__init__()
        self.shape  = [size, size]
        self.distribution = np.zeros(self.shape)                              

    def createSimulateObject(self, path):
        self.distribution = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), self.shape, interpolation=cv2.INTER_CUBIC).astype(np_float_datatype)


class PhaseObject3D():
    def __init__(self, shape, pad, n_media, n_min, n_max):
        assert len(shape) == 3
        self.shape       = shape
        self.pad         = pad
        self.n_media     = n_media
        self.n_min       = n_min
        self.n_max       = n_max
        self.RI          = np.ones(shape, dtype = np_float_datatype)
        self.RI_pad      = np.ones((shape[0]+2*pad, shape[1]+2*pad, shape[2]), dtype = np_float_datatype)
        self.RI_cuda     = torch.ones(shape).to(config.device)
        self.RI_pad_cuda = torch.ones((shape[0]+2*pad, shape[1]+2*pad, shape[2])).to(config.device)


    def createPhaseObject3DfromSlice(self, slice_dic):
        i = 0
        for slice_name, slice in slice_dic.items():
            if slice.distribution.shape[0] == config.slice_size:
                self.RI[..., i] = slice.distribution
                self.RI_pad = self.__pad(self.RI, self.pad)
            elif slice.distribution.shape[0] == config.slice_size:
                self.RI_pad[..., i] = slice.distribution
                self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
            i += 1
        RI = self.RI
        RI_max = np.max(RI)
        RI_min = np.min(RI)
        RI = (RI-RI_min)/(RI_max-RI_min)
        # self.RI = (self.n_max-self.n_media) / RI_max * (RI-RI_max) + self.n_max - self.n_media
        self.RI = (self.n_max-self.n_media) * (RI-1) + self.n_max - self.n_media
        self.RI_pad = self.__pad(self.RI, config.pad_size)
        self.__array2tensorRI()

    def createPhaseObject3DfromTensorDic(self, tensor_dic): # 由3DTensor对PhaseObject做初始化，发生在优化过程中，tv正则化后
        i = 0
        for slice_name, slice in tensor_dic.items():
            if slice.shape[0] == config.slice_size:
                self.RI_cuda[..., i] = slice
                self.RI_pad_cuda = self.__pad_torch(self.RI_cuda, self.pad)
            elif slice.shape[0] == config.slice_pad_size:
                self.RI_pad_cuda[..., i] = slice
                self.RI_cuda = self.RI_pad_cuda[self.pad:-self.pad, self.pad:-self.pad]
            else:
                raise('The shape of slice_dic error!')
            i += 1
        if self.pad != 0:
            self.RI_cuda = self.RI_pad_cuda[self.pad:-self.pad, self.pad:-self.pad]
        else:
            self.RI_cuda = self.RI_pad_cuda
        self.__tensor2arrayRI()

    def createPhaseObject3DfromArrayDic(self, array_dic): # 由3DArray对PhaseObject做初始化，发生在优化发生前
        i = 0
        for slice_name, slice in array_dic.items():
            if slice.shape[0] == config.slice_size:
                self.RI[..., i] = slice
                self.RI_pad = self.__pad(self.RI, self.pad)
            elif slice.shape[0] == config.slice_pad_size:
                self.RI_pad[..., i] = slice
                self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
            else:
                raise('The shape of slice_dic error!')
            i += 1
        self.__array2tensorRI()
    
    def createPhaseObject3Dfrom3DArray(self, array_3d):
        RI = array_3d
        RI_max = np.max(RI)
        RI_min = np.min(RI)
        RI = (RI-RI_min)/(RI_max-RI_min)
        if RI.shape[0] == config.slice_size:
            self.RI = (self.n_max-self.n_media) * (RI-1) + self.n_max - self.n_media
            self.RI_pad = self.__pad(self.RI, self.pad)
        elif RI.shape[0] == config.slice_pad_size:
            self.RI_pad = (self.n_max-self.n_media) * (RI-1) + self.n_max - self.n_media
            self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
        self.__array2tensorRI()

    def createPhaseObject3Dfrom2DArray(self, array_2d):
        RI_max = np.max(array_2d)
        RI_min = np.min(array_2d)
        RI_2d = (array_2d-RI_min)/(RI_max-RI_min)
        RI_2d = (self.n_max-self.n_media) * (RI_2d-1) + self.n_max - self.n_media
        if array_2d.shape[0] == config.slice_size:
            for i in range(config.slice_num_list[-1]):
                self.RI[:, :, i] = RI_2d
                self.RI_pad = self.__pad(self.RI, self.pad)
        elif array_2d.shape[0] == config.slice_pad_size:
            for i in range(config.slice_num_list[-1]):
                self.RI_pad[:, :, i] = RI_2d
                self.RI = self.RI_pad[self.pad:-self.pad, self.pad:-self.pad]
        else:
            raise('The shape of array_2d error!')
        self.__array2tensorRI()

    def zeroInitPhaseObject3D(self):
        for i in range(self.RI.shape[2]):
            self.RI[..., i] = np.zeros([config.slice_size, config.slice_size])
            self.RI_cuda[..., i] = torch.zeros([config.slice_size, config.slice_size])
            self.RI_pad[..., i] = np.zeros([config.slice_pad_size, config.slice_pad_size])
            self.RI_pad_cuda[..., i] = torch.zeros([config.slice_pad_size, config.slice_pad_size])
    
    def uniformInitPhaseObject3D(self):
        rd = np.random.RandomState(888)
        trd_RI = torch.randn(config.slice_size, config.slice_size)
        trd_RI_pad = torch.randn(config.slice_pad_size, config.slice_pad_size)
        for i in range(self.RI.shape[2]):
            self.RI[..., i] = rd.uniform(0, config.n_max-config.n_media, (config.slice_size, config.slice_size))
            self.RI_cuda[..., i] = trd_RI.uniform_(0, config.n_max-config.n_media)
            self.RI_pad[..., i] = rd.uniform(0, config.n_max-config.n_media, (config.slice_pad_size, config.slice_pad_size))
            self.RI_pad_cuda[..., i] = trd_RI.uniform_(0, config.n_max-config.n_media)

    def createBackgroundObject(self):
        self.RI = np.zeros(self.RI.shape)
        self.RI_pad = np.zeros(self.RI_pad.shape)

    def __array2tensorRI(self):
        self.RI_cuda = torch.tensor(self.RI)
        self.RI_pad_cuda = torch.tensor(self.RI_pad)
    
    def __tensor2arrayRI(self):
        self.RI = self.RI_cuda.cpu().detach().numpy()
        self.RI_pad = self.RI_pad_cuda.cpu().detach().numpy()

    def __pad(self, obj, pad_size):
        return np.pad(obj, ((pad_size, pad_size), (pad_size, pad_size),  (0,0)))

    def __pad_torch(self, obj, pad_size):
        return F.pad(obj, (pad_size, pad_size, pad_size, pad_size, 0, 0))
    
    def saveObjectAsNpy(self):
        np.save('./3d_sample.npy', self.RI)

    def showObject(self):
        plt.figure()
        plt.subplot(231),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]/2 - 4)], vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid-2')
        plt.subplot(232),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]/2 + 0)], vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid')
        plt.subplot(233),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]/2 + 2)], vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+2')
        plt.subplot(234),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]/2 + 4)], vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+4')
        plt.subplot(235),  plt.imshow(cv2.resize(self.RI[:,145,:], (120, self.RI.shape[0])), vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid')
        plt.subplot(236),  plt.imshow(cv2.resize(self.RI[90,:,:],  (120, self.RI.shape[0])), vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid')
        plt.colorbar()

        plt.figure()
        plt.subplot(231),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 - 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid-2')
        plt.subplot(232),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 0)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid')
        plt.subplot(233),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 2)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+2')
        plt.subplot(234),  plt.imshow(self.RI[:,:,round(self.RI.shape[2]//2 + 4)], cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xy mid+4')
        plt.subplot(235),  plt.imshow(cv2.resize(self.RI[:,145,:], (120, self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid')
        # plt.subplot(235),  plt.imshow(cv2.resize(self.RI.shape[1]//2, (round(config.imaging_depth/config.HR_pixel_size), self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid') # 将z轴显示尺寸拉伸到实际比例，显示中间层
        plt.subplot(236),  plt.imshow(cv2.resize(self.RI[90,:,:],  (120, self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('yz mid')
        # plt.subplot(236),  plt.imshow(cv2.resize(self.RI.shape[0]//2, (round(config.imaging_depth/config.HR_pixel_size), self.RI.shape[0])), cmap='gray', vmin=config.plot_range[0], vmax=config.plot_range[1]), plt.title('xz mid') # 将z轴显示尺寸拉伸到实际比例，显示中间层


    def saveOptimObject(self, epoch):
        root_path = './optim_process'
        path_3d = os.path.join(root_path, '3d_results')
        
        if   config.scatter_mode == 'MS':
            if config.tv:
                path_3d = os.path.join(path_3d, 'MS-tv')
            else:
                path_3d = os.path.join(path_3d, 'MS')
        elif config.scatter_mode == 'MS_LIAC':
            if config.tv:
                path_3d = os.path.join(path_3d, 'MS_LIAC-tv')
            else:
                path_3d = os.path.join(path_3d, 'MS_LIAC')
        elif config.scatter_mode == 'MLB':
            if config.tv:
                path_3d = os.path.join(path_3d, 'MLB-tv')
            else:
                path_3d = os.path.join(path_3d, 'MLB')
                
        if os.path.exists(os.path.join(path_3d,'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))) == False:
            os.mkdir(os.path.join(path_3d, 'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1])))
        path_3d = os.path.join(path_3d, 'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))

        if os.path.exists('epoch{}'.format(epoch)) == False:
            os.mkdir(os.path.join(path_3d, 'epoch{}'.format(epoch)))
        path_3d = os.path.join(path_3d, 'epoch{}'.format(epoch))

        for i in range(config.slice_num_list[-1]):
            slice = self.RI[:, :, i]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xy_{}.jpg'.format(i)))


    def saveObject(self, loss_list):
        # self.showObject()
        root_path = './experiment_results'
        path_2d = os.path.join(root_path, '2d_results')
        path_3d = os.path.join(root_path, '3d_results')

        if   config.scatter_mode == 'MS':
            if config.tv:
                path_2d = os.path.join(path_2d, 'MS-tv')
                path_3d = os.path.join(path_3d, 'MS-tv')
            else:
                path_2d = os.path.join(path_2d, 'MS')
                path_3d = os.path.join(path_3d, 'MS')
        elif config.scatter_mode == 'MS_LIAC':
            if config.tv:
                path_2d = os.path.join(path_2d, 'MS_LIAC-tv')
                path_3d = os.path.join(path_3d, 'MS_LIAC-tv')
            else:
                path_2d = os.path.join(path_2d, 'MS_LIAC')
                path_3d = os.path.join(path_3d, 'MS_LIAC')
        elif config.scatter_mode == 'MLB':
            if config.tv:
                path_2d = os.path.join(path_2d, 'MLB-tv')
                path_3d = os.path.join(path_3d, 'MLB-tv')
            else:
                path_2d = os.path.join(path_2d, 'MLB')
                path_3d = os.path.join(path_3d, 'MLB')

        plt.savefig(os.path.join(path_2d,'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}].jpg'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1])))
        
        if os.path.exists(os.path.join(path_3d,'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))) == False:
            os.mkdir(os.path.join(path_3d, 'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1])))
        path_3d = os.path.join(path_3d, 'epoch{}_lr{}_led{}_imaging_depth{}_tv{}-{}-{}_plt_range[{},{}]'.format(config.num_epoch,config.learning_rate, config.led_mode, config.imaging_depth, config.tv_beta[0], config.tv_beta[1], config.tv_beta[2], config.plot_range[0], config.plot_range[1]))
        for i in range(config.slice_num_list[-1]):
            slice = self.RI[:, :, i]
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xy_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            slice = cv2.resize(self.RI[:, i, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'xz_{}.jpg'.format(i)))

        for i in range(config.slice_size):
            slice = cv2.resize(self.RI[i, :, :], (self.RI.shape[2]*2, self.RI.shape[0]))
            slice_im = (slice-(config.plot_range[0]))/(config.plot_range[1]-(config.plot_range[0])) * 255
            im = Image.fromarray(slice_im)
            im = im.convert('L')
            im.save(os.path.join(path_3d, 'yz_{}.jpg'.format(i)))    
        ### 保存loss_list ###
        # filename = open(os.path.join(path_3d, 'epoch{}_lr{}.jpg'.format(config.num_epoch,config.learning_rate)), 'w')
        # for value in loss_list:
        #     filename.write(str(value)+'\n')
        # filename.close()
        ### ------------ ###
        plt.show()
        

class LedArray():
    def __init__(self):
        super(LedArray, self).__init__()
        self.loc            = []
        self.kxky           = []
        self.fxfy           = []
        self.numxy_center   = []
    
    def createLedLocation(self):
        N       = config.arraysize # N代表LED矩阵中一条灯带的灯数
        gap     = config.led_gap
        delta_x = config.x_offset
        delta_y = config.y_offset
        theta   = config.theta_offset

        xlocation = np.zeros([N ** 2])
        ylocation = np.zeros([N ** 2])
        for i in range(1, N+1):
            xlocation[N*(i-1):N+N*(i-1)] = np.arange(-(N-1)/2, N/2, 1)*gap
            ylocation[N*(i-1):N+N*(i-1)] = ((N-1)/2-(i-1))*gap
        x_location = (xlocation * np.cos(theta) + ylocation * np.sin(theta) + delta_x)
        y_location = (xlocation * np.sin(theta) - ylocation * np.cos(theta) + delta_y)
        return x_location, y_location
    
    def generateIlluminationSequence(self):
        N = config.arraysize
        n = (N + 1)/2
        sequence = np.zeros([2, N**2])
        sequence[0, 0] = n
        sequence[1, 0] = n
        dx = 1
        dy = -1
        stepx = 1
        stepy = -1
        direction = 1
        counter = 0
        i = 1
        while i < N**2:
            counter += 1
            if direction == 1:
                sequence[0, i] = sequence[0, i-1] + dx
                sequence[1, i] = sequence[1, i-1]
                if counter == abs(stepx):
                    counter = 0
                    direction = direction * -1
                    dx = dx * -1
                    stepx = stepx * -1
                    if stepx > 0:
                        stepx += 1
                    else:
                        stepx -= 1
            else:
                sequence[0, i] = sequence[0, i-1]
                sequence[1, i] = sequence[1, i-1] + dy
                if counter == abs(stepy):
                    counter = 0
                    direction = direction * -1
                    dy = dy * -1
                    stepy = stepy * -1
                    if stepy > 0:
                        stepy += 1
                    else:
                        stepy -= 1
            i += 1
        illumination_sequence = (sequence[0, :] - 1) * N + sequence[1, :]
        return illumination_sequence

    def createIlluminationAngle(self):
        N    = config.num_illu
        h    = config.led_height
        k0   = config.k0
        x, y = self.createLedLocation()
        seq  = self.generateIlluminationSequence()
        kx_relative = - np.sin(np.arctan(x / h))
        ky_relative = - np.sin(np.arctan(y / h))
        kx = k0 * kx_relative
        ky = k0 * ky_relative
        fx = k0 * kx_relative / (2*pi)
        fy = k0 * ky_relative / (2*pi)
        test = np.zeros([N,2])
        for idx, value in enumerate(seq):
            value = int(value) - 1
            self.loc.append([x[value], y[value]])
            self.kxky.append([kx[value], ky[value]])
            self.fxfy.append([fx[value], fy[value]])
            test[idx,0] = fx[value]
            test[idx,1] = fy[value]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:')
        plt.show()
        return
    
    def generateSpiralPath(self, revs):
        N = config.num_illu
        t = np.linspace(0, 2*pi, N)
        x = t * np.cos(revs*t)/(2*pi)
        y = t * np.sin(revs*t)/(2*pi)
        dist =  np.zeros(N)
        for i in range(1, N):
            dist[i] = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) + dist[i-1]
        coef = np.mean(t[N-1]**2/dist[N-1])
        tInc = np.sqrt(np.linspace(0, dist[N-1], N+1)*coef)
        tInc = tInc[0:N]
        x_ = tInc*np.cos(revs*tInc)/(2*pi)
        y_ = tInc*np.sin(revs*tInc)/(2*pi)
        x = config.NA/config.wlength * x_
        y = config.NA/config.wlength * y_
        test = np.zeros([N,2])
        for i in range(N):
            self.kxky.append([2*pi*y[i], 2*pi*x[i]])
            self.fxfy.append([y[i], x[i]])
            test[i,0] = y[i]
            test[i,1] = x[i]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:')
        # plt.show()
        self.createLocfromRead()
        return 0
    
    def readkxky(self):
        param = io.loadmat(config.param_path)
        test = np.zeros([config.num_illu,2])
        for i in range(config.num_illu):
            self.kxky.append([-2*pi*param['kx'][0,i], -2*pi*param['ky'][0,i]])
            self.fxfy.append([-param['kx'][0,i], -param['ky'][0,i]])
            # self.fxfy.append([param['kx'][0,i], param['ky'][0,i]])
            test[i,0] = -param['kx'][0,i]
            test[i,1] = -param['ky'][0,i]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.show()
        return

    def readArraykxky(self):
        feature = h5py.File('./Parameters.mat')  #读取mat文件
        kx = feature['kx'][0:config.num_illu]
        ky = feature['ky'][0:config.num_illu]
        test = np.zeros([config.num_illu,2])
        for i in range(config.num_illu):
            if i < kx.shape[0]:
                self.kxky.append([kx[i,0], ky[i,0]])
                self.fxfy.append([kx[i,0]/(2*pi*1e6), ky[i,0]/(2*pi*1e6)])
                test[i,0] = self.fxfy[i][0]
                test[i,1] = self.fxfy[i][1]
        plt.figure()
        plt.plot(test[:,0], test[:,1], 'o:'), plt.title('illumination angle')
        plt.xlabel('fx')
        plt.ylabel('fy')
        plt.show()
        return

    def createLocfromRead(self):
        wlength = config.wlength
        for i in range(config.num_illu):
            if i < len(self.kxky):
                loc_x = np.tan(np.arcsin(wlength * self.fxfy[i][0])) * config.led_height
                loc_y = np.tan(np.arcsin(wlength * self.fxfy[i][1])) * config.led_height
                self.loc.append([loc_x, loc_y])
            else:
                print('The number of capture img is out of the num_illu!')
        return


class Measurement():
    def __init__(self, num_measurement):
        self.size = num_measurement
        self.show_measurement = list([x,x+1,x+2] for x in range(self.size))        
        self.in_measurement   = list([x,x+1,x+2] for x in range(self.size))  
        self.exposure_coeff   = np.ones(config.num_illu)

    def in2show_transform(self):
        for i in range(self.size):
            self.show_measurement[i] = self.in_measurement[i].to('cpu').detach().numpy()

    def show2in_transform(self):
        for i in range(self.size):
            self.in_measurement[i] = torch.tensor(self.show_measurement[i]).to(config.device)
    
    def readMatMeasurement(self):
        path = './Potato_raw_image.mat'
        feature = h5py.File(path)  #读取mat文件
        # data = feature['imageSeq'][0:config.num_illu, 1000:2800, 500:2300]  #读取mat文件中所有数据存储到array中
        data = feature['imageSeq'][0:config.num_illu, 1600:2200, 1100:1700]
        for i in range(data.shape[0]):
            if i < len(self.show_measurement):
                self.show_measurement[i] = data[i]

    def readCEMeasurement(self):
        img = sitk.ReadImage(config.data_path)
        img = sitk.GetArrayFromImage(img)
        # img = np.resize(img, (img.shape[0], config.slice_size, config.slice_size))
        for i in range(img.shape[0]):
            if i < len(self.show_measurement):
                self.show_measurement[i] = img[i, 560:920, 0:360] # Phary
                # self.show_measurement[i] = img[i, 360:720, 780:1140] # mouth
                # self.show_measurement[i] = img[i] # total FOV
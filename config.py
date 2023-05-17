from math import pi
import torch
import numpy as np

bit = 32
np_float_datatype   = np.float32 if bit == 32 else np.float64
np_complex_datatype = np.complex64 if bit == 32 else np.complex128
torch_float_datatype = torch.float32 if bit == 32 else torch.float64
torch_complex_datatype = torch.complex64 if bit == 32 else torch.complex128

import os
import warnings
warnings.filterwarnings("ignore")

def initDevice(use_gpu=False):
        torch.set_printoptions(precision=8)
        if use_gpu:
            print('cuda availabel is {}'.format(torch.cuda.is_available()))
            return 'cuda'
        else:
            return 'cpu'

use_gpu = True
device  = initDevice(use_gpu)

class Configs:
    def __init__(self):
        self.np_float       = np_float_datatype
        self.torch_float    = torch_float_datatype
        self.np_complex     = np_complex_datatype
        self.torch_complex  = torch_complex_datatype

        self.device         = device
        self.is_simulation  = False
        self.led_mode       = 'spiral'
        self.scatter_mode   = 'MS_LIAC' # 'MS','MS_LIAC','MLB'   
        self.tv             = True
        self.tv_beta        = [1, 0.05, 0.5] #[1, 0.05, 0.5]
        self.en_optim_light = False

        self.capture_size        = 360
        self.slice_size          = 360
        self.pad_size            = 30   # each side
        self.slice_pad_size      = self.slice_size + 2*self.pad_size
        self.slice_num_list      = [120]
        self.phase_obj_shape     = (self.slice_size, self.slice_size, self.slice_num_list[-1])
        self.phase_obj_pad_shape = (self.slice_pad_size, self.slice_pad_size, self.slice_num_list[-1])
        self.plot_range          = [-0.03, 0.07]


        self.imaging_depth = 30   # 
        self.n_media       = 1.33 # refractive index of immersion media
        self.n_min         = 1.33 # refractive index of min density feature
        self.n_max         = 1.51 # refractive index of max density feature


        self.data_path = './FOV_01_reconstruction_CElegan_resources/FOV_01_rawdata_CElegan.tif'
        self.param_path = './FOV_01_reconstruction_CElegan_resources/FOV_01_reconstruction_CElegan_params.mat'
            

        self.en_optim_object   = True
        self.learning_rate     = 4.8e-4 # actual 4.8e-4 5e-4 1e-3
        self.num_epoch         = 50
        self.wlength           = 0.532 # um green
        self.k0                = 2*pi / self.wlength
        self.NA                = 1.0556 #1.0556
        self.HR_pixel_size     = 0.1154 #0.1154 # Waller
        self.imaging_width     = self.slice_size* self.HR_pixel_size

        self.__num_illu   = 120
        self.arraysize    = 11
        self.num_illu     = self.arraysize**2 if self.led_mode == 'array' else self.__num_illu
        self.led_gap      = 5    # mm
        self.led_height   = 100  # mm
        self.x_offset     = 0    # mm
        self.y_offset     = 0    # mm
        self.theta_offset = 0    # mm

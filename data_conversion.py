import scipy.io as io
import h5py
# import SimpleITK as sitk
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

from config import Configs

config = Configs()

def create_ROI():
	data_path = './data/2USAFchart_10x_473nm'
	img = cv2.imread(os.path.join(data_path, '0.jpg'))
	cv2.namedWindow('sphere', flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
	cv2.imshow('sphere', img)
	roi = cv2.selectROI('sphere', img, showCrosshair=True, fromCenter=False)
	x, y, w, h = roi
	print(roi)
	rect = config.slice_size
	cv2.rectangle(img=img, pt1=(round(x+w/2)-rect//2, round(y+h/2)-rect//2), pt2=(round(x+w/2)+rect//2, round(y+h/2)+rect//2), color=(255,255,255), thickness=2)
	img_roi = img[round(y+h/2)-rect//2:round(y+h/2)+rect//2, round(x+w/2)-rect//2:round(x+w/2)+rect//2]
	cv2.imshow('sphere_roi', img_roi)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for i in range(289):
		img = cv2.imread(os.path.join(data_path, '{}.jpg'.format(i)))
		im = np.array(img)
		im_roi = im[round(y+h/2)-rect//2:round(y+h/2)+rect//2, round(x+w/2)-rect//2:round(x+w/2)+rect//2]
		cv2.imwrite(os.path.join('./data/2USAFchart_10x_473nm/ROI', '{}.jpg'.format(i)), im_roi)
	return


def npy2mat():
	np_array = np.load('./3d_sample_1000epoch_.npy')
	io.savemat('./3d_sample_1000epoch_.mat',{'data':np_array})


# root_path = './data/3D_potato'
# mat_path = os.path.join(root_path, 'Potato_raw_image.mat')
# npy_path = os.path.join(root_path, 'Potato_raw_image.npy')
# npy_roi_path = os.path.join(root_path, 'Potato_raw_image_roi.npy')

root_path = './data/singlelayer_bloodsmear'
mat_path = os.path.join(root_path, 'Bloodsmear_raw_image.mat')
npy_path = os.path.join(root_path, 'Bloodsmear_raw_image.npy')
npy_roi_path = os.path.join(root_path, 'Bloodsmear_raw_image_roi.npy')

def readmat():
	mat = h5py.File(mat_path, 'r')
	# print(mat.keys())
	# print(mat.values())
	img = mat['imageSeq'][:]
	np.save(npy_path, img)

	# max = np.max(img)
	# min = np.min(img)
	# for i in range(img.shape[0]):
	# 	image = img[i, ...]
	# 	cv2.imwrite(os.path.join(root_path, 'ori_int8', '{}.jpg'.format(i)), image)
	# 	image = (image - min) / (max-min) * 255
	# 	cv2.imwrite(os.path.join(root_path, 'high_contrast_int8', '{}.jpg'.format(i)), image)
	# return


def create_ROI_2():
	img = cv2.imread(os.path.join(root_path, 'high_contrast_int8', '0.jpg'))
	cv2.namedWindow('totalFOV', flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
	cv2.imshow('totalFOV', img)
	roi = cv2.selectROI('totalFOV', img, showCrosshair=True, fromCenter=False)
	x, y, w, h = roi
	print(roi)
	rect = config.slice_size
	cv2.rectangle(img=img, pt1=(round(x+w/2)-rect//2, round(y+h/2)-rect//2), pt2=(round(x+w/2)+rect//2, round(y+h/2)+rect//2), color=(255,255,255), thickness=2)
	img_roi = img[round(y+h/2)-rect//2:round(y+h/2)+rect//2, round(x+w/2)-rect//2:round(x+w/2)+rect//2]
	cv2.imshow('roi', img_roi)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	imgs = np.load(npy_path)
	np.save(npy_roi_path, imgs[:, round(y+h/2)-rect//2:round(y+h/2)+rect//2, round(x+w/2)-rect//2:round(x+w/2)+rect//2])

	max = np.max(imgs)
	min = np.min(imgs)

	for i in range(imgs.shape[0]):
		im = imgs[i]
		im = (im-min) / (max-min) * 255
		im_roi = im[round(y+h/2)-rect//2:round(y+h/2)+rect//2, round(x+w/2)-rect//2:round(x+w/2)+rect//2]
		cv2.imwrite(os.path.join(root_path, 'high_contrast_ROI_int8', '{}.jpg'.format(i)), im_roi)
	return


if __name__ == '__main__':
		# create_ROI()
		npy2mat()
		# readmat()
		# create_ROI_2()

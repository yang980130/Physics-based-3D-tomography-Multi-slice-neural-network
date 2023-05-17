# Physics-based 3D tomography Multi-slice neural network(MSNN)

torch(python) implementation of paper: **3D refractive index imaging with an untrained physical-based multi-slice Neural Network**. This GitHub provides the codes, and links to test data.

## citation

If you use this project in your own research, please cite our corresponding paper:

Delong Yang, Shaohui Zhang, Yao Hu and Qun Hao, **3D refractive index imaging with an untrained physical based multi-slice Neural Network**

## Abstract

we demonstrate an untrained physics-based 3D tomography multi-slice neural network (MSNN), in which each layer has a clear corresponding physical meaning according to the beam propagation model.
The network does not require pre-training and performs good generalization and can be recovered through the optimization of a set of actually acquired intensity images.
Concurrently, MSNN can calibrate the intensity of different illumination and the multiple backscattering effects have also been taken into consideration by asserting a "scattering attenuation layer" between adjacent "refractive index" layers in the MSNN. The experiments have been conducted carefully to demonstrate the effectiveness and feasibility of the proposed method.

## Requirement 

**numpy, pytorch(Minimum version 1.7.0), cuda11.1**,opencv-python, matplotlib, scipy.

## Usage

Run **experiment.py** to run the MSNN for C.elegan 3D tomography, the data of C.elegan is from  [Professor Laura Waller's Lab](https://drive.google.com/drive/folders/19eQCMjTtiK8N1f1nGtXlfXkEa8qL6kDl). If you want to use this code to reconstruct your own data, please carefully adjust the parameters in config.py to fit your optical system configuration. 

## Result

[result](The reconstruction results with TV regularization.pdf)

## License

This project is licensed under the terms of the BSD-3-Clause license.
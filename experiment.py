from config import Configs
from model import PhaseObject3D, LedArray, Measurement
from nn_model import MultiSliceNN, generate_optimizer, Optimization

import matplotlib.pyplot as plt


import warnings

def experiment():

    config          = Configs()
    source          = LedArray()
    model           = MultiSliceNN(slice_num = config.slice_num_list[-1]).to(config.device)
    optimizer, scheduler = generate_optimizer(model, config.learning_rate)
    optim           = Optimization(model, optimizer, scheduler)
    init_object_3d  = PhaseObject3D(config.phase_obj_shape, config.pad_size, config.n_media, config.n_min, config.n_max)
    
    gt_measurement = Measurement(config.num_illu)
    gt_measurement.readCEMeasurement()
    gt_measurement.show2in_transform()
    show_measurement = gt_measurement.show_measurement

    plt.figure()
    plt.imshow(show_measurement[0] , cmap='gray')
    plt.figure()
    plt.subplot(221), plt.imshow(show_measurement[90] , cmap='gray')
    plt.subplot(222), plt.imshow(show_measurement[100] , cmap='gray')
    plt.subplot(223), plt.imshow(show_measurement[110] , cmap='gray')
    plt.subplot(224), plt.imshow(show_measurement[120] , cmap='gray')
    plt.show()

    in_measurement = gt_measurement.in_measurement
    show_measurement = gt_measurement.show_measurement
 
    init_object_3d.zeroInitPhaseObject3D()
    source.readkxky()
    
    model.initModel(init_object_3d)
    optim.train(source, in_measurement)
    recover_sample = model.extractParameters2cpu()
    # recover_sample.save3DObjectAsNpy()
    recover_sample.showObject()
    recover_sample.saveObject(optim.loss_list)
    plt.show()
    return


def main():
    warnings.filterwarnings("ignore")
    experiment()


if __name__ == "__main__":
    main()

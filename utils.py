import os
import yaml
import math
import numpy as np

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# def normalization(cood,axi_x_min,axi_x_max,axi_y_min,axi_y_max,axi_z_min,axi_z_max):
#     cood[:, 0] = (cood[:,0]-axi_x_min)/(axi_x_max-axi_x_min)
#     cood[:, 1] = (cood[:,1]-axi_y_min)/(axi_y_max-axi_y_min)
#     cood[:, 2] = (cood[:,2]-axi_z_min)/(axi_z_max-axi_z_min)
#     return cood

def normalization(cood, x_min, x_max, y_min, y_max, z_min, z_max):
    mins = np.array([x_min, y_min, z_min])
    maxs = np.array([x_max, y_max, z_max])
    return (cood - mins) / (maxs - mins)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(log_directory):
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)

    return checkpoint_directory, image_directory,log_directory


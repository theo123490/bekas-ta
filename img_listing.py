import glob
import numpy as np


def image_list(filename) :
    img_list = glob.glob(filename + '/*.jpg')
    img_list = np.asarray(img_list)
    img_list = [i.split(filename + '\\')[1] for i in img_list]
    return img_list

"""
script to load the labelled images
imgs_lablled = no_images X no_cells X no_of objects
no_objects = 4 where 0 = ice, 1 = wood, 2=stone, 3 = pig

action = list of actions corresponnding to each image
"""


import numpy as np
import pickle
import pdb

data_path = "/home/meghna/projSAILON/"

with open(data_path+'labelled_data.pickle', 'rb') as f:
    data = pickle.load(f)
    imgs_labelled = data["labelled_imgs"]
    action = data["actions"]
    tap = data["tap_times"]
    file = data["img_name"]
    pdb.set_trace()

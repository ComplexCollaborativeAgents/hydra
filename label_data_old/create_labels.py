import numpy as np
import os
import pickle
import cv2
import pdb
import matplotlib.pyplot as plt


def load_unlablled_data(file):
    """
    loads the unlabelled data file
    """

    with open(file,"rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(e)
    img_prev = data['img_prev']
    symb_prev = data['symb_prev']
    action = data['action']
    symb_next = data['symb_next']
    tap_time = data['tap']

    return img_prev, symb_prev, action, symb_next, tap_time

def load_labelled_date(file):
    with open(data_path+'labelled_data.pickle', 'rb') as f:
        data = pickle.load(f)
        imgs_labelled = data["labels"]
        action = data["actions"]
        tap = data["tap_times"]
        file_name = data["img_name"]
    return imgs_labelled, action, tap, file_name


def get_center_pixels(symb, idx):
    """
    finds the center vertices of the symbolic object
    """
    vertices = symb[idx]['vertices']
    center_r = 0
    center_c = 0
    for v in range(0,len(vertices)):
        center_r += (480 - vertices[v]['y'])
        center_c += vertices[v]['x']
    center_r = int(center_r/len(vertices))
    center_c = int(center_c/len(vertices))

    return [center_r, center_c]

def fix_pixel_coords(coords,r_crop,c_crop,r_curr,c_curr, offset_r, offset_c):
    """
    function converts the pixels coordinates from original images to the
    cropped and resized image
    """

    coord_r = ((coords[0] - offset_r)/r_crop)*r_curr
    coord_c = ((coords[1] - offset_c)/c_crop)*c_curr

    return [int(coord_r),int(coord_c)]

def find_cell(desc_factor_r, desc_factor_c,pixel,grid_r):
    """
    given pixel coords finds the corresponding cell of the descretised 84X84 image
    """
    no_rows = grid_r
    pixels_r = np.floor(pixel[0]/desc_factor_r)
    pixels_c = np.floor(pixel[1]/desc_factor_c)
    cell_no = (grid_r-1)*pixels_r + pixels_c

    return int(cell_no)


def color_match(color_obj, cmap):
    """
    it checks if the colormap of the object matches with ice/wood/stone/pig
    """
    if len(color_obj) >= len(cmap) + 4 or len(color_obj) <= len(cmap) - 4:
        return False
    if len(np.setdiff1d(color_obj,cmap)) < 4:
        return True
    return False

def label_symb(symb):
    offset_r = 80
    offset_c = 20
    r_crop = 310
    c_crop = 770
    r_curr = 84
    c_curr = 84
    grid_r = 7
    grid_c = 7
    desc_factor_r = r_curr/grid_r
    desc_factor_c = c_curr/grid_c
    no_cells = (grid_r)*(grid_c)
    no_objects = 4   ## 0 = ice, 1 = wood, 2=stone, 3 = pig
    objects = ['ice', 'wood', 'stone', 'pig']
    ## colormap of the symbolic objects
    ice = np.asarray([19,55,87,123,91,51,119,255,155,187,191,223,136])
    wood = np.asarray([19,136,73,168,172,204,240,208,245,244,25,132,21])
    stone = np.asarray([19,73,109,146,77,182,219,114,136])
    pig  = np.asarray([0,4,40,44,8,48,84,88,120,121,125,124,80,116,52,113,117,152,186,219,255,223,76,148,218,150,36,112,149,28,146,182,227,73,109,31,251,97,161,89])
    object_cmapping = {'ice':ice, 'wood':wood,'stone':stone,'pig':pig}

    labels = np.zeros((no_cells, no_objects))
    for i in range(0,len(symb)):
        if symb[i]['type'] == 'Object':
            color = symb[i]['colormap']
            color_obj = []
            for j in range(0,len(color)):
                color_obj.append(color[j]['color'])

            color_obj =  np.asarray(color_obj)
            for obj_idx in range(0,no_objects):
                cmap = object_cmapping[objects[obj_idx]]
                if color_match(color_obj,cmap):
                    # pdb.set_trace()
                    center_coords = get_center_pixels(symb,i)
                    center_coords_remapped = fix_pixel_coords(center_coords,r_crop,c_crop,r_curr,c_curr, offset_r, offset_c)
                    cell_no = find_cell(desc_factor_r,desc_factor_c,center_coords_remapped,grid_r)
                    labels[cell_no,obj_idx]+= 1
    return labels

def label_dataset(data_path):
    data_entries = os.listdir(data_path)

    no_images = len(data_entries)
    labels = []
    img_list = []
    symb_list = []
    action_list = []
    tap_times = []
    file_name = []
    img_no = -1
    for file in data_entries:
        img_prev, symb_prev, action, symb_next, tap_time = load_unlablled_data(data_path+file)
        img_list.append(img_prev)
        symb_list.append(label_symb(symb_prev))
        action_list.append(action)
        tap_times.append(tap_time)
        file_name.append(file)
        img_no+=1
        print(img_no)
        labels.append(label_symb(symb_next))

    images = np.stack(img_list)
    symbolic = np.stack(symb_list)
    labels = np.stack(labels)
    tap_times = np.array(tap_times)
    action_list = np.array(action_list)
    return images, symbolic, labels, action_list, tap_times, file_name

if __name__ == '__main__':

    
    ## path to the data directory
    data_path = "data/train/"
    images, symbolic, labels, action_list, tap_times, file_name = label_dataset(data_path)
    train = {'images': images, 'symbolic': symbolic, 'labels': labels,'actions':action_list, 'tap_times': tap_times, 'img_name':file_name}
    data_path = "data/test/"
    images, symbolic, labels, action_list, tap_times, file_name = label_dataset(data_path)
    test = {'images': images, 'symbolic': symbolic, 'labels': labels,'actions':action_list, 'tap_times': tap_times, 'img_name':file_name}    

    with open('labelled_data.pickle', 'wb') as f:
            pickle.dump({'train': train, 'test': test}, f)

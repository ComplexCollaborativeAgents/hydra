import numpy as np
import torch
from shapely.geometry import Polygon
import cv2
from PIL import Image

from agent.perception.novelty_detection.model_cosine import ModelCosine
import torchvision.transforms as transforms

import pickle
import matplotlib.pyplot as plt

import pdb
class ObsToState():

    def Unpack_ScienceBirdsObservation(self,obs):
        sb_state = obs.state  # SBState
        sb_action = obs.action # SBAction
        sb_intermediate_states = obs.intermediate_states
        reward = obs.reward

        return sb_state, sb_action, sb_intermediate_states, reward

    def Unpack_SBState(self, sb_state):
        objects = sb_state.objects
        image = sb_state.image
        game_state = sb_state.game_state
        sling = sb_state.sling

        return objects, image, game_state, sling

    def Obs_to_StateImage(self, obs):
        sb_state, sb_action, sb_intermediate_states, reward = self.Unpack_ScienceBirdsObservation(obs)
        state, image , _ , _ = self.Unpack_SBState(sb_state)

        return state, image


class NoveltyDetector():
    def __init__(self, model_path, class_info_path):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.thresholds = np.load(class_info_path)['thresholds']

        self.class_list = np.load(class_info_path)['class_list']
        print(self.class_list)
        self.novelty_model = ModelCosine(len(self.class_list))


        self.novelty_model.load_state_dict(torch.load(model_path, map_location=self._device))

        self.weight_ll = self.novelty_model.classifier[4].weight.detach().cpu().numpy()
        self.img_dim = (32,32)
        # self.image = image
        # self.state = state
    def init_state(self,state, image):
        self.image = image
        self.state = state

    def evalaute(self):
        """
        returns dict with type and novelty
        """
        keys = list(self.state.keys())
        subImgs = []
        novelty_dict = {}
        id = 0
        for k in range(len(keys)):
            object = self.state[keys[k]]
            type = object['type']
            poly = object['polygon']
            if poly.type == "Polygon":
                # print(type)
                subImg = self.img_to_subImg(type, poly)
                activations_ll = self.extract_activations(subImg)
                novelty, type_predicted, sim_scores = self.detect_novelty(activations_ll)
                novelty_dict[keys[k]] = {'novelty':novelty, 'type':type, 'predicted_type':type_predicted, 'sim_scores':sim_scores}

                print(novelty, type, type_predicted, sim_scores)
                ## do novelty detection
            elif poly.type == "MultiPolygon":
                for p in range(len(poly)):
                    subImg = self.img_to_subImg(type, poly[p])
                    activations_ll = self.extract_activations(subImg)
                    novelty, type_predicted, sim_scores = self.detect_novelty(activations_ll)
                    novelty_dict[keys[k]] = {'novelty':novelty, 'type':type, 'predicted_type':type_predicted, 'sim_scores':sim_scores}
                    # print(novelty, type, type_predicted)
                ## do novelty detection

    def extract_activations(self,subImg):
        trans = transforms.Compose([transforms.ToTensor()])
        subImg_tensor = trans(subImg)
        subImg_tensor_reshaped =  subImg_tensor.reshape((1,3,32,32))
    
        # subImg_tensor = torch.tensor(subImg)
        self.novelty_model.eval()
        with torch.no_grad():
            layers, out = self.novelty_model(subImg_tensor_reshaped)

        act = layers[len(layers)-2].cpu().numpy()
        return act

    def detect_novelty(self,act):

        scores = np.matmul(act,self.weight_ll.T)[0]

        max_score = np.max(scores)
        max_score_class = np.argmax(scores)
        # pdb.set_trace()
        if max_score > self.thresholds[max_score_class]:
            novelty = False
            return novelty, self.class_list[max_score_class], scores
        novelty = True
        # pdb.set_trace()
        return novelty, self.class_list[max_score_class], scores


    def img_to_subImg(self,type, poly):

        x, y = poly.exterior.coords.xy
        poly_coords = np.int32(np.vstack((x,y)).T)

        x_min = np.min(poly_coords[:,0])
        x_max = np.max(poly_coords[:,0])
        y_min = np.min(poly_coords[:,1])
        y_max = np.max(poly_coords[:,1])
        subImg = self.image[y_min:y_max,x_min:x_max,:]

        saveImg = Image.fromarray(subImg)
        saveImg.save("tmp.png")
        subImg = cv2.imread("tmp.png")
        subImg = cv2.resize(subImg,self.img_dim)
        # pdb.set_trace()
        # plt.imshow(subImg)

        # plt.show()
        return subImg

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:58:27 2019

@author: chengxue
"""
import sys
sys.path.append('..')
sys.path.append('./src')
import numpy as np
#import os
import cv2
import json
from computer_vision.game_object import GameObject, GameObjectType
from computer_vision.cv_utils import Rectangle
import logging
from  shapely.geometry import Polygon

class NotVaildStateError(Exception):
   """NotVaildStateError exceptions"""
   pass

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ground_truth")



class GroundTruthReader:
    def __init__(self, json, model, target_class, new_level=True):

        '''
        json : a list of json objects. the first element is int id, 2nd is png sreenshot
        if sreenshot is required, and the rest of them is the ground truth of game
        objects

        look_up_matrix: matrix of size n * 256, where n is the number of tempelet we used, 256 represents the 8bit color value

        look_up_obj_type: length n array, storing the type corrsponding to the look_up_matrix

        '''
        self.model = model
        self.target_class = target_class

        self.type_transformer = {
                'bird_blue':'blueBird',
                'bird_yellow':'yellowBird',
                'bird_black':'blackBird',
                'bird_red':'redBird',
                'bird_white':'whiteBird',
                'platform':'hill',
                'pig_basic_big' : 'pig',
                'pig_basic_small' : 'pig',
                'pig_basic_medium' : 'pig',
                'TNT' : 'TNT',
                'Slingshot':'slingshot',
                'ice' : 'ice',
                'stone' : 'stone',
                'wood' : 'wood',
                'unknown' : 'unknown'
                }

        self.contour_color = {
                'bird_blue': (189,160,79)[::-1],
                'bird_yellow':(56,215,240)[::-1],
                'bird_black': (0,0,0)[::-1],
                'bird_red': (40,0,218,)[::-1],
                'bird_white':(200,200,200)[::-1],
                'Platform':(200,200,200)[::-1],
                'pig_basic_big' : (67,225,78)[::-1],
                'pig_basic_small' : (67,225,78)[::-1],
                'pig_basic_medium' : (67,225,78)[::-1],
                'TNT' : (58,113,194)[::-1],
                'Slingshot':(48,102,160)[::-1],
                'Ice' : (224,200,96)[::-1],
                'Stone' : (150,150,150)[::-1],
                'Wood' : (31,117,210)[::-1]
                }


        #combine the platforms
        #replace exsiting platforms with new combined platforms
        self.alljson = []
        json = json[0]['features']
        for j in json:
             if True or j['properties']['label'] != 'Platform':
                 self.alljson.append(j)

        self._parseJsonToGameObject()

#        if not self.is_vaild():
#            raise NotVaildStateError('request new state')
        self.new_level = new_level


    def is_vaild(self):
        '''
        check if the stats received are vaild or not

        for vaild state, there has to be at least one pig and one bird.
        '''

        pigs = self.find_pigs_mbr()
        birds = self.find_birds()

        if pigs and birds:
            return True
        else:
            return False


    def set_screenshot(self, screenshot):
        self.screenshot = screenshot

    def _parseJsonToGameObject(self):
        '''
        convert json objects to game objects
        '''

        self.allObj = {}

        #find the type of all object

        #1. vectorize the dictionary of colors
        obj_num = 0
        obj_total_num = len(self.alljson)
        obj_matrix = np.zeros((256,obj_total_num))
        obj_types = np.zeros(obj_total_num).astype(str)

        for j in self.alljson:
            if j['properties']['label'] == "Slingshot" or j['properties']['label'] == "Ground" or j['properties']['label'] == "Trajectory":
                obj_types[obj_num] = j['properties']['label']

            else:
                colorMap = j['properties']['colormap']
                for pair in colorMap:
                    obj_matrix[int(float(pair['color']))][obj_num] = pair['percent']

            obj_num += 1

        #model predict
        intercept = self.model[:,0].reshape(-1,1)
        model_coef = self.model[:,1:]

        predicts = intercept + model_coef @ obj_matrix
        predict_class = predicts.argmax(0)
        predict_class = np.array(list(map(lambda x : self.target_class[x],predict_class)))
        obj_types[obj_types=='0.0'] =  predict_class[obj_types=='0.0']

        #put unknow for the dot products less than a threshold

        obj_num = 0
        for j in self.alljson:

            if j['properties']['label'] == "Slingshot":


                rect = self._getRect(j)
                contours = j['geometry']['coordinates']
                vertices = contours[0]

                game_object = GameObject(rect,GameObjectType(self.type_transformer["Slingshot"]),vertices)

                try:
                    self.allObj[self.type_transformer["Slingshot"]].append(game_object)
                except:
                    self.allObj[self.type_transformer["Slingshot"]] = [game_object]

            elif j['properties']['label'] == "Ground" or j['properties']['label'] == "Trajectory":
                pass

            else:
#                logger.info("class {},{}".format(obj_types[obj_num],max(predicts[:,obj_num])))
                rect = self._getRect(j)
                contours = j['geometry']['coordinates']
                vertices = contours[0]
                game_object = GameObject(rect,GameObjectType(self.type_transformer[obj_types[obj_num]]),vertices)

                try:
                    self.allObj[self.type_transformer[obj_types[obj_num]]].append(game_object)
                except:
                    self.allObj[self.type_transformer[obj_types[obj_num]]] = [game_object]
            obj_num += 1






    def _getRect(self,j):
        '''
        input: json object
        output: rectangle of the object
        '''
        contours = j['geometry']['coordinates']
        vertices = contours[0]

        x = []
        y = []
        for v in vertices:
            x.append(int(float(v[0])))
            y.append(int(float(v[1])))
        points = (np.array(y),np.array(x))
        rect = Rectangle(points)
        return rect


    def find_bird_on_sling(self,birds,sling):
        sling_top_left = sling.top_left[1]
        distance = {}
        for bird_type in birds:
            if len(birds[bird_type]) > 0:
                for bird in birds[bird_type]:
                    #print(bird)
                    distance[bird] = abs(bird.top_left[1]\
                                    - sling_top_left)
        min_distance = 1000
        for bird in distance:
            if distance[bird] < min_distance:
                ret = bird
                min_distance = distance[bird]
        return ret

    def find_hill_mbr(self):
        ret = self.allObj.get('hill',None)
        return ret

    def find_pigs_mbr(self):
        ret = self.allObj.get('pig',None)
        return ret

    def find_platform_mbr(self):
        ret = self.allObj.get('Platform',None)
        return ret

    def find_slingshot_mbr(self):
        ret = self.allObj.get('slingshot',None)
        return ret

    def find_birds(self):
        ret = {}
        for key in self.allObj:
            if 'Bird' in key:
                ret[key] = self.allObj[key]
        if len(ret) == 0:
            return None
        else:
            return ret

    def find_blocks(self):
        ret = {}
        for key in self.allObj:
            if 'wood' in key or 'ice' in key or 'stone' in key or 'TNT' in key:
                ret[key] = self.allObj[key]
        if len(ret) == 0:
            return None
        else:
            return ret

    def showResult(self):
        '''
        draw the ground truth result
        '''

        contours = []
        contour_types = []
        for obj in self.alljson:
            if obj['type'] == 'Ground':
                y_index = int(float(obj['yindex']))
            else:
                #create contours
                contour = np.zeros((len(obj['vertices']),1,2))
                for i in range(len(obj['vertices'])) :
                    contour[i,:,0] = obj['vertices'][i]['x']
                    contour[i,:,1] = obj['vertices'][i]['y']

                contours.append(contour.astype(int))
                contour_types.append(obj['type'])

        #return contours

        for i in range(len(contours)):
            cv2.drawContours(self.screenshot, contours, i , self.contour_color[contour_types[i]],1)
            cv2.putText(self.screenshot,contour_types[i],
                    tuple(tuple(np.min(contours[i],0)[0])),
                    0,
                    0.3,
                    (255,0,0))

        cv2.line(self.screenshot, (0,y_index), (839,y_index), (0,255,0), 1)
        cv2.imshow('ground truth',self.screenshot[:,:,::-1])
        cv2.waitKey(30)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    model = np.loadtxt("model",delimiter=",")
    target_class = list(map(lambda x : x.replace("\n", ""),open('target_class').readlines()))

    result = json.load(open("../../0_GTData.json",'r'))

    gt = GroundTruthReader(result,model,target_class)

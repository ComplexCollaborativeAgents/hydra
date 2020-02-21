#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:58:27 2019

@author: chengxue
"""
import numpy as np
import os
import cv2
import json
from computer_vision.game_object import GameObject, GameObjectType
from computer_vision.cv_utils import Rectangle

class NotVaildStateError(Exception):
   """NotVaildStateError exceptions"""
   pass

class GroundTruthReader:
    def __init__(self,json, ignore_invalid_state = False):

        '''
        json : a list of json objects. the first element is int id, 2nd is png sreenshot
        if sreenshot is required, and the rest of them is the ground truth of game
        objects
        '''

        #print('reading json')

#        self.shape_transformer = {
#                    "RectTiny": GameObjectShape.RectTiny,
#                    "RectSmall": GameObjectShape.RectSmall,
#                    "Rect": GameObjectShape.Rect,
#                    "RectMedium": GameObjectShape.RectMedium,
#                    "RectBig": GameObjectShape.RectBig,
#                    "RectFat": GameObjectShape.RectFat,
#
#                    "SquareTiny": GameObjectShape.SquareTiny,
#                    "SquareSmall": GameObjectShape.SquareSmall,
#                    "Square": GameObjectShape.Square,
#                    "SquareBig": GameObjectShape.SquareBig,
#                    "SquareHole": GameObjectShape.SquareHole,
#
#                    "Triangle": GameObjectShape.Triangle,
#                    "TriangleHole": GameObjectShape.TriangleHole,
#
#                    "CircleSmall": GameObjectShape.CircleSmall,
#                    "Circle": GameObjectShape.Circle
#                }
        # use top3 color to identify an object
#        self.color_distribution_table = {(73, 0, 32): 'bird_black',
#                                         (73, 32, 0): 'bird_black',
#                                         (164,): 'bird_black',
#                                         (36, 0, 109): 'bird_black',
#                                         (64, 100): 'bird_black',
#                                         (73, 0, 36): 'bird_black',
#                                         (73, 36, 0): 'bird_black',
#                                         (119, 0, 36): 'bird_blue',
#                                         (119, 0, 82): 'bird_blue',
#                                         (119, 0, 41): 'bird_blue',
#                                         (0, 250, 244): 'bird_red',
#                                         (36, 21, 255): 'bird_white',
#                                         (36, 255, 21): 'bird_white',
#                                         (248, 36): 'bird_yellow',
#                                         (248, 0): 'bird_yellow',
#                                         (248,): 'bird_yellow',
#                                         (248, 0, 36): 'bird_yellow',
#                                         (36, 68): 'Platform',
#                                         (205, 68, 172): 'TNT',
#                                         (19, 87, 123): 'Ice',
#                                         (19, 123, 87): 'Ice',
#                                         (123, 19, 87): 'Ice',
#                                         (123, 87, 19): 'Ice',
#                                         (19, 87, 255): 'Ice',
#                                         (19, 255, 87): 'Ice',
#                                         (124, 156, 19): 'Ice',
#                                         (19, 123, 255): 'Ice',
#                                         (19, 255, 123): 'Ice',
#                                         (87, 19, 123): 'Ice',
#                                         (87, 123, 19): 'Ice',
#                                         (19, 123, 155): 'Ice',
#                                         (19, 155, 123): 'Ice',
#                                         (19, 155, 136): 'Ice',
#                                         (125, 36, 4): 'pig_basic_big',
#                                         (125, 84): 'pig_basic_small',
#                                         (125, 203): 'pig_basic_medium',
#                                         (125, 203, 4): 'pig_basic_medium',
#                                         (125, 36, 84): 'pig_basic_big',
#                                         (125, 84, 80): 'pig_basic_medium',
#                                         (125,): 'pig_basic_small',
#                                         (125, 36): 'pig_basic_medium',
#                                         (125, 121): 'pig_basic_small',
#                                         (125, 84, 121): 'pig_basic_small',
#                                         (125, 84, 36): 'pig_king',
#                                         (73, 19, 182): 'Stone',
#                                         (146, 73, 182): 'Stone',
#                                         (146, 182, 73): 'Stone',
#                                         (182, 73, 146): 'Stone',
#                                         (182, 146, 73): 'Stone',
#                                         (73, 146, 182): 'Stone',
#                                         (73, 182, 146): 'Stone',
#                                         (136, 172, 168): 'Wood',
#                                         (136, 240, 172): 'Wood',
#                                         (136, 240, 245): 'Wood',
#                                         (136, 240, 208): 'Wood',
#                                         (136, 245, 240): 'Wood',
#                                         (136, 172, 240): 'Wood',
#                                         (136, 245, 172): 'Wood',
#                                         (240, 136, 204): 'Wood',
#                                         (136, 240, 19): 'Wood',
#                                         (240, 136, 172): 'Wood',
#                                         (136, 73, 240): 'Wood',
#                                         (136, 240, 73): 'Wood'}

        self.color_distribution_table = {(73, 0, 32): 'bird_black',
                                         (73, 32, 0): 'bird_black',
                                         (164, 36, 64): 'bird_black',
                                         (64, 100, 0): 'bird_black',
                                         (36, 0, 109): 'bird_black',
                                         (73, 0, 36): 'bird_black',
                                         (73, 36, 0): 'bird_black',
                                         (119, 0, 36): 'bird_blue',
                                         (119, 0, 82): 'bird_blue',
                                         (119, 0, 41): 'bird_blue',
                                         (0, 250, 244): 'bird_red',
                                         (165, 128, 250): 'bird_red_big',
                                         (36, 21, 255): 'bird_white',
                                         (36, 255, 21): 'bird_white',
                                         (248, 36, 72): 'bird_yellow',
                                         (248, 36, 212): 'bird_yellow',
                                         (248, 0, 36): 'bird_yellow',
                                         (248, 36, 0): 'bird_yellow',
                                         (146, 219, 182): 'effects_',
                                         (32, 255, 68): 'effects_',
                                         (146, 182, 255): 'effects_',
                                         (182, 255, 146): 'effects_',
                                         (36, 255, 219): 'effects_',
                                         (172, 146, 32): 'effects_',
                                         (205, 68, 172): 'TNT',
                                         (36, 68, 32): 'Platform',
                                         (19, 87, 123): 'Ice',
                                         (123, 87, 19): 'Ice',
                                         (19, 123, 87): 'Ice',
                                         (123, 19, 87): 'Ice',
                                         (19, 87, 255): 'Ice',
                                         (19, 255, 87): 'Ice',
                                         (124, 156, 19): 'Ice',
                                         (87, 123, 19): 'Ice',
                                         (87, 19, 123): 'Ice',
                                         (19, 123, 255): 'Ice',
                                         (19, 255, 123): 'Ice',
                                         (19, 123, 155): 'Ice',
                                         (19, 155, 123): 'Ice',
                                         (19, 155, 136): 'Ice',
                                         (125, 203, 84): 'pig_basic_medium',
                                         (125, 36, 4): 'pig_basic_medium',
                                         (125, 84, 88): 'pig_basic_small',
                                         (125, 203, 4): 'pig_basic_big',
                                         (125, 36, 84): 'pig_basic_medium',
                                         (125, 84, 203): 'pig_basic_big',
                                         (125, 36, 88): 'pig_basic_medium',
                                         (125, 84, 80): 'pig_basic_medium',
                                         (125, 121, 84): 'pig_basic_small',
                                         (125, 84, 36): 'pig_basic_small',
                                         (125, 84, 121): 'pig_basic_small',
                                         (125, 84, 0): 'pig_basic_small',
                                         (73, 19, 182): 'Stone',
                                         (146, 73, 182): 'Stone',
                                         (146, 182, 73): 'Stone',
                                         (182, 73, 146): 'Stone',
                                         (73, 182, 146): 'Stone',
                                         (182, 146, 73): 'Stone',
                                         (73, 146, 182): 'Stone',
                                         (136, 240, 172): 'Wood',
                                         (136, 240, 245): 'Wood',
                                         (240, 136, 172): 'Wood',
                                         (136, 172, 168): 'Wood',
                                         (136, 240, 208): 'Wood',
                                         (136, 245, 240): 'Wood',
                                         (136, 172, 240): 'Wood',
                                         (136, 73, 240): 'Wood',
                                         (136, 240, 19): 'Wood',
                                         (240, 136, 204): 'Wood',
                                         (136, 245, 172): 'Wood',
                                         (136, 240, 73): 'Wood'}

        self.type_transformer = {
                'bird_blue':'blueBird',
                'bird_yellow':'yellowBird',
                'bird_black':'blackBird',
                'bird_red':'redBird',
                'bird_white':'whiteBird',
                'Platform':'hill',
                'pig_basic_big' : 'pig',
                'pig_basic_small' : 'pig',
                'pig_basic_medium' : 'pig',
                'TNT' : 'TNT',
                'Slingshot':'slingshot',
                'Ice' : 'ice',
                'Stone' : 'stone',
                'Wood' : 'wood',
                'Unknown': 'unknown'
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
#        self.platformCombied = platformCombiner(json)

        #replace exsiting platforms with new combined platforms
        self.alljson = []
        for j in json:
            if j['type'] != 'Platform':
                self.alljson.append(j)
#        self.alljson.extend(self.platformCombied)

        self._parseJsonToGameObject()

        if not (self._if_vaild() or ignore_invalid_state):
            raise NotVaildStateError('request new state')


    def _if_vaild(self):
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
        for j in self.alljson:
            #prepare rectangle

            if j['type'] == "Slingshot":


                rect = self._getRect(j)
                vertices = j["vertices"]

                game_object = GameObject(rect,vertices,GameObjectType(self.type_transformer["Slingshot"]))

                try:
                    self.allObj[self.type_transformer["Slingshot"]].append(game_object)
                except:
                    self.allObj[self.type_transformer["Slingshot"]] = [game_object]

            elif j['type'] == "Ground" or j['type'] == "Trajectory":
                pass

            else:
                #print(j['colormap'])
                rect = self._getRect(j)
                vertices = j["vertices"]


                #use color map to decide the object type

                colorMap = j['colormap']
                color_list = []
                for color in colorMap:
                    colorCleaned = [0,0]
                    colorCleaned[0] = int(color['color'])
                    colorCleaned[1] = float(color['percent'])

                    color_list.append((int(colorCleaned[0]),float(colorCleaned[1])))
                top_3_color = sorted(color_list,key = lambda x : x[1], reverse = True)[:3]
                top_3_color = tuple([x[0] for x in top_3_color])

                game_object = GameObject(rect,vertices,GameObjectType(self.type_transformer.get(self.color_distribution_table.get(top_3_color,'Unknown'),'unknown')))

                try:
                    self.allObj[self.type_transformer[self.color_distribution_table.get(top_3_color,'Unknown')]].append(game_object)
                except:
                    self.allObj[self.type_transformer[self.color_distribution_table.get(top_3_color,'Unknown')]] = [game_object]



    def _getRect(self,j):
        '''
        input: json object
        output: rectangle of the object
        '''
        vertices = j['vertices']
        x = []
        y = []
        for v in vertices:
            x.append(int(v['x']))
            y.append(480 - int(v['y']))
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
                y_index = 480 - int(obj['yindex'] )
            else:
                #create contours
                contour = np.zeros((len(obj['vertices']),1,2))
                for i in range(len(obj['vertices'])) :
                    contour[i,:,0] = obj['vertices'][i]['x']
                    contour[i,:,1] = 480 - obj['vertices'][i]['y']

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

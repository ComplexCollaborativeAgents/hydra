#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:58:27 2019

@author: chengxue
"""
import sys
sys.path.append('./science_birds_interface')


import numpy as np
import os
import cv2
import json
from computer_vision.game_object import GameObject, GameObjectType,GameObjectShape
from computer_vision.cv_utils import Rectangle, platformCombiner

class NotVaildStateError(Exception):
   """NotVaildStateError exceptions"""
   pass

class GroundTruthReader:
    def __init__(self,json):

        '''
        json : a list of json objects. the first element is int id, 2nd is png sreenshot
        if sreenshot is required, and the rest of them is the ground truth of game
        objects
        '''

        self.shape_transformer = {
                    "RectTiny": GameObjectShape.RectTiny,
                    "RectSmall": GameObjectShape.RectSmall,
                    "Rect": GameObjectShape.Rect,
                    "RectMedium": GameObjectShape.RectMedium,
                    "RectBig": GameObjectShape.RectBig,
                    "RectFat": GameObjectShape.RectFat,

                    "SquareTiny": GameObjectShape.SquareTiny,
                    "SquareSmall": GameObjectShape.SquareSmall,
                    "Square": GameObjectShape.Square,
                    "SquareBig": GameObjectShape.SquareBig,
                    "SquareHole": GameObjectShape.SquareHole,

                    "Triangle": GameObjectShape.Triangle,
                    "TriangleHole": GameObjectShape.TriangleHole,

                    "CircleSmall": GameObjectShape.CircleSmall,
                    "Circle": GameObjectShape.Circle
                }

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
                'Wood' : 'wood'
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
        self.platformCombied = platformCombiner(json)

        #replace exsiting platforms with new combined platforms
        self.alljson = []
        for j in json:
            if j['type'] != 'Platform':
                self.alljson.append(j)
        self.alljson.extend(self.platformCombied)

        self._parseJsonToGameObject()
        
        if not self._if_vaild():
            raise NotVaildStateError('request new state')
        
    
    def _if_vaild(self):
        '''
        check if the stats received are vaild or not
        
        for vaild state, there has to be at least one pig and one bird and a slingshot.
        '''

        pigs = self.find_pigs_mbr()
        birds = self.find_birds()
        sling = self.find_slingshot_mbr()
                    
        if pigs and birds and sling:
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
            object_type = j['type']
            if 'vertices' in j:
                vertices = j['vertices']
                x = []
                y = []
                for v in vertices:
                    x.append(int(v['x']))
                    y.append(480 - int(v['y']))
                points = (np.array(y),np.array(x))
                rect = Rectangle(points)
                if 'name' in j:
                    ##remove bracket - ad-hoc, will be fixed
                    if '(' in j['name']:
                        j['name'] = j['name'][:j['name'].find('(')]
                    game_object = GameObject(rect,type = GameObjectType(self.type_transformer[object_type]), vertices = vertices,
                                             shape = self.shape_transformer[j['name']])
                else:
                    game_object = GameObject(rect,GameObjectType(self.type_transformer[object_type]),vertices,
                                             shape = GameObjectShape.Rect) # for the first release: use only the rectangle

                try:
                    self.allObj[self.type_transformer[object_type]].append(game_object)
                except:
                    self.allObj[self.type_transformer[object_type]] = [game_object]

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
        drew the ground truth result
        '''
        try:
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
        except:
            print('Errors in displaying ground truth.')
            pass



if __name__ == "__main__":
    root = "/Users/chengxue/Gitlab/sciencebirds/"
    paths = os.listdir(root)
    f = open(root + 'PigData.json','r')
    result = json.load(f)
    gt = GroundTruthReader(result)
#    gt.showResult()
#    import matplotlib.pylab as plt
#    aa =[j for j in gt.alljson if j['type'] == 'Ice']
#    x = [597,597,597,597,629,629,598]
#    y = [317,317,295,294,294,295,317]
#    y = [480 - i for i in y]
#    plt.scatter(x,y)

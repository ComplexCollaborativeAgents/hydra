import random
import os
import settings
#from agent.perception import perception
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn import preprocessing
import csv
import pandas as pd

OBJECT_CLASSES = {'bird_black_1': 'blackBird',
               'bird_black_2': 'blackBird',
               'bird_white_1': 'birdWhite',
               'bird_white_2': 'birdWhite',
               'bird_yellow_1': 'yellowBird',
               'bird_yellow_2': 'yellowBird',
               'bird_blue_1': 'blueBird',
               'bird_blue_2': 'blueBird',
               'bird_red_1': 'redBird',
               'bird_red_2': 'redBird',
               'platform': 'hill',
               'ice_triang_1': 'ice', 'ice_square_hole_1': 'ice', 'ice_triang_hole_1': 'ice',
               'ice_rect_small_1': 'ice',
               'ice_square_small_1': 'ice', 'ice_rect_tiny_1': 'ice', 'ice_rect_big_1': 'ice',
               'ice_rect_fat_1': 'ice',
               'ice_rect_medium_1': 'ice', 'ice_circle_1': 'ice', 'ice_square_tiny_1': 'ice',
               'ice_circle_small_1': 'ice',
               'wood_rect_big_1': 'wood', 'wood_rect_tiny_1': 'wood', 'wood_rect_tiny_2': 'wood',
               'wood_circle_small_1': 'wood',
               'wood_square_hole_1': 'wood', 'wood_rect_small_1': 'wood', 'wood_square_small_1': 'wood',
               'wood_triang_hole_1': 'wood', 'wood_circle_1': 'wood', 'wood_rect_medium_1': 'wood',
               'wood_square_tiny_1': 'wood', 'wood_square_small_2': 'wood', 'wood_rect_fat_1': 'wood',
               'wood_triang_1': 'wood',
               'stone_rect_fat_1': 'stone', 'stone_rect_medium_1': 'stone', 'stone_circle_1': 'stone',
               'stone_square_small_1': 'stone', 'stone_rect_small_1': 'stone', 'stone_rect_big_1': 'stone',
               'stone_square_tiny_2': 'stone',
               'stone_square_hole_1': 'stone', 'stone_rect_tiny_1': 'stone', 'stone_triang_1': 'stone', 'stone_rect_fat_2': 'stone', 'stone_rect_big_2': 'stone',
               'stone_triang_2': 'stone', 'stone_square_small_2': 'stone',
               'stone_circle_small_1': 'stone', 'stone_square_tiny_1': 'stone', 'stone_triang_hole_1': 'stone',
               'stone_triang_hole_2': 'stone',
               'stone_rect_tiny_2': 'stone',
               'stone_circle_small_2': 'stone',
               'pig_basic_medium_3': 'pig', 'pig_basic_medium_1': 'pig', 'pig_basic_small_1': 'pig', 'pig_basic_small_3': 'pig', 'pig_basic_small_6': 'pig',
               'Slingshot': 'slingshot', 'TNT': 'TNT','Platform':'platform',
               'worm': 'worm',
               'magician': 'magician',
               'wizard': 'wizard',
               'butterfly': 'butterfly'}


OBJECT_CLASSES_EDITED = {
    'bird_black': 'blackBird',
    'bird_white': 'birdWhite',
    'bird_yellow': 'yellowBird',
    'bird_blue': 'blueBird',
    'bird_red': 'redBird',
    'platform': 'hill',
    'ice_triang': 'ice', 'ice_square_hole': 'ice', 'ice_triang_hole': 'ice',
    'ice_rect_small': 'ice',
    'ice_square_small': 'ice', 'ice_rect_tiny': 'ice', 'ice_rect_big': 'ice',
    'ice_rect_fat': 'ice',
    'ice_rect_medium': 'ice', 'ice_circle': 'ice', 'ice_square_tiny': 'ice',
    'ice_circle_small': 'ice',
    'wood_rect_big': 'wood', 'wood_rect_tiny': 'wood',
    'wood_circle_small': 'wood',
    'wood_square_hole': 'wood', 'wood_rect_small': 'wood', 'wood_square_small': 'wood',
    'wood_triang_hole': 'wood', 'wood_circle': 'wood', 'wood_rect_medium': 'wood',
    'wood_square_tiny': 'wood', 'wood_square_small': 'wood', 'wood_rect_fat': 'wood',
    'wood_triang': 'wood',
    'stone_rect_fat': 'stone', 'stone_rect_medium': 'stone', 'stone_circle': 'stone',
    'stone_square_small': 'stone', 'stone_rect_small': 'stone', 'stone_rect_big': 'stone',
    'stone_square_tiny': 'stone',
    'stone_square_hole': 'stone', 'stone_rect_tiny': 'stone', 'stone_triang': 'stone', 'stone_rect_fat': 'stone', 'stone_rect_big': 'stone',
    'stone_circle_small': 'stone', 'stone_square_tiny': 'stone', 'stone_triang_hole': 'stone',
    'stone_triang_hole': 'stone',
    'stone_rect_tiny': 'stone',
    'stone_circle_small': 'stone',
    'pig_basic_medium': 'pig', 'pig_basic_small': 'pig', 'pig_basic_big': 'pig',
    'Slingshot': 'slingshot', 'TNT': 'TNT','Platform':'platform',
    'worm': 'worm',
    'magician': 'magician',
    'wizard': 'wizard',
    'butterfly': 'butterfly'
}

def type_to_class(type):
    classes = OBJECT_CLASSES
    if type in classes:
        return classes[type]
    else:
        print(type)
        assert 'novel' in type
        return type

def object_class_convert(obj_label: str):
    """
    Converts raw data object classes into general object classes (ie black_bird_7 -> blackBird)
    """
    if obj_label in OBJECT_CLASSES_EDITED:
        return OBJECT_CLASSES_EDITED[obj_label]
    else:
        trunc_label = obj_label[:-2] # Truncate the _<number> end part of the object class
        if trunc_label not in OBJECT_CLASSES_EDITED:
            # raise KeyError("Encountered unexpected object label: {} (truncated to {})".format(obj_label, trunc_label))
            return "unknown"
        return OBJECT_CLASSES_EDITED[trunc_label]


def train_classifier(file=os.path.join(settings.ROOT_PATH,'data/science_birds/perception/pIII/non_novel_objects.csv'), on_full_data=False):
    #reader = csv.DictReader(open(file, 'r'), perception.classification_cols())
    df = pd.read_csv(file)
    print(len(df.iloc[:, 0]))
    y = [type_to_class(x) for x in df.iloc[:, 0]]
    x = df.iloc[:, 1:]
    logreg = lm.LogisticRegression(max_iter=500000, multi_class='ovr')
    if not on_full_data:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        logreg.fit(x_train.values, y_train)
        predictions = logreg.predict(x_test.values)
        print(classification_report(y_test, predictions))
    else:
        logreg.fit(x.values,y)
    return logreg
# probably need some preprocessing

def test_classifier(logreg,file = os.path.join(settings.ROOT_PATH,'data/science_birds/perception/object_class_level_1.csv')):
    df = pd.read_csv(file)
    performance=[]
    for row in df.to_numpy():
        #print(row)
        type = type_to_class(row[0])
        prediction = logreg.predict_proba([row[1:]])
        proposal = logreg.classes_[prediction[0].argmax()]
        probability = max(prediction[0])
        if probability > settings.SB_CLASSIFICATION_THRESHOLD:
            pred_decision = proposal
        else:
            pred_decision = 'unknown'
        if pred_decision=='unknown' or 'novel' in type or proposal != type:
            performance.append({'env_type':type,
                                'proposal': proposal,
                                'probability': probability,
                                'pred_decision': pred_decision,
                               })
    return performance



def test_object_type_prediction(logreg, input_vector):
    prediction = logreg.predict_proba(input_vector)
    proposal = logreg.classes_[prediction[0].argmax()]
    probability = max(prediction[0])
    print(proposal, probability)


if __name__ == '__main__':
    import pickle

    #### train/load
    # logreg = train_classifier(on_full_data=True)
    # pickle.dump(logreg, open('{}/data/science_birds/perception/logreg_pIII.p'.format(settings.ROOT_PATH), 'wb'))


    ### test
    logreg = pickle.load(open('{}/data/science_birds/perception/logreg_pIII.p'.format(settings.ROOT_PATH), 'rb'))
    performance = test_classifier(logreg, file=os.path.join(settings.ROOT_PATH,'data/science_birds/perception/pIII/novel_object_type22.csv'))
    #performance = test_classifier(logreg, file=os.path.join(settings.ROOT_PATH,'data/science_birds/perception/pIII/novel_object_level11_type50.csv'))
    #performance= test_classifier(logreg, file=os.path.join(settings.ROOT_PATH,'data/science_birds/perception/pII/object_class.csv'))
    #performance = test_classifier(logreg, file=os.path.join(settings.ROOT_PATH,'data/science_birds/perception/pII/non_novel_objects.csv'))

    for item in performance:
        print(item)


#    pickle.dump(logreg,open('{}/data/science_birds/perception/logreg.p'.format(settings.ROOT_PATH), 'wb'))